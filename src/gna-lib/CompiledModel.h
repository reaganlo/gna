/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "AccelerationDetector.h"
#include "HardwareCapabilities.h"
#include "HardwareModelScorable.h"
#include "SoftwareModel.h"
#include "SubModel.h"

namespace GNA
{

class CompiledModel
{
public:
    template<class T>
    CompiledModel(
        const T & model,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn,
        std::vector<std::unique_ptr<Memory>>& memoryObjects) :
        LayerCount{ GetNumberOfOperations(model) },
        GmmCount{ getGmmCount(GetFirstOperation(model), LayerCount) },
        detector{ detectorIn },
        hwCapabilities{ hwCapabilitiesIn },
        memoryList{ memoryObjects },
        softwareModel
        {
            model,
            makeValidator(),
            detector.GetSupportedCpuAccelerations()
        }
    {
    }

    virtual ~CompiledModel() = default;
    CompiledModel(const CompiledModel &) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;

    void BuildHardwareModel(DriverInterface &ddi);

    const std::vector<std::unique_ptr<Layer>>& GetLayers() const;
    const Layer* GetLayer(uint32_t layerIndex) const;

    void AddUniqueMemory(Memory *memory);
    Memory * FindBuffer(const void *buffer, const size_t bufferSize) const;
    void IdentifyBuffer(const void *buffer, size_t bufferSize);
    uint32_t CalculateSize() const;
    void CopyData(void *address, size_t size) const;

    void InvalidateConfig(gna_request_cfg_id configId,
            LayerConfiguration *layerConfiguration, uint32_t layerIndex) const;

    bool IsPartOfModel(Memory *memory) const;

    Gna2Status Score(
        RequestConfiguration& config,
        RequestProfiler *profiler,
        KernelBuffers *buffers);

    void ValidateBuffer(std::vector<Memory *> &configMemoryList, Memory *memory) const;

    const uint32_t LayerCount;
    const uint32_t GmmCount;
    const std::vector<Memory *>& GetModelMemoryList() const
    {
        return modelMemoryList;
    }

protected:
    std::unique_ptr<HardwareModelScorable> hardwareModel;

private:
    const std::vector<std::unique_ptr<SubModel>>&
        getSubmodels(const HardwareCapabilities& hwCaps);

    void createSubmodels(const HardwareCapabilities& hwCaps);

    SubmodelType getSubmodelType(
            const HardwareCapabilities &hwCaps, const Layer& layer) const;

    uint32_t scoreAllSubModels(RequestConfiguration& config,
        RequestProfiler *profiler, KernelBuffers *buffers);

    BaseValidator makeValidator();
    static uint32_t GetNumberOfOperations(const Gna2Model& model)
    {
        return model.NumberOfOperations;
    }
    static uint32_t GetNumberOfOperations(const gna_model& model)
    {
        return model.nLayers;
    }
    static Gna2Operation* GetFirstOperation(const Gna2Model& model)
    {
        return model.Operations;
    }
    static intel_nnet_layer_t* GetFirstOperation(const gna_model& model)
    {
        return model.pLayers;
    }
    static bool isGmmOperation(const nn_layer& layer)
    {
        return layer.operation == INTEL_GMM;
    }
    static bool isGmmOperation(const Gna2Operation& operation)
    {
        return operation.Type == Gna2OperationTypeGmm;
    }

    template<class T>
    uint32_t getGmmCount(const T* firstOperation, uint32_t numberOfOperations)
    {
        uint32_t gmmCount = 0;
        for (uint32_t i = 0; i < numberOfOperations; i++)
        {
            if (isGmmOperation(firstOperation[i]))
            {
                ++gmmCount;
            }
        }
        return gmmCount;
    }

    const AccelerationDetector& detector;
    const HardwareCapabilities& hwCapabilities;
    std::vector<std::unique_ptr<Memory>>& memoryList;
    std::vector<Memory *> modelMemoryList;
    SoftwareModel softwareModel;
    std::map<DeviceVersion,
            std::vector<std::unique_ptr<SubModel>>> submodels;

};

}
