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
    CompiledModel(
        const gna_model *const userModel,
        const AccelerationDetector &detectorIn,
        std::vector<std::unique_ptr<Memory>>& memoryObjects);

    virtual ~CompiledModel() = default;
    CompiledModel(const CompiledModel &) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;

    void BuildHardwareModel(DriverInterface &ddi, HardwareCapabilities &hwCaps);

    const std::vector<std::unique_ptr<Layer>>& GetLayers() const;
    const Layer* GetLayer(uint32_t layerIndex) const;
    decltype(auto) GetSubmodels() const
    {
        return (submodels);
    }

    void AddUniqueMemory(Memory *memory);
    Memory * FindBuffer(const void *buffer, const size_t bufferSize) const;
    void IdentifyBuffer(const void *buffer, size_t bufferSize);
    uint32_t CalculateSize() const;
    void CopyData(void *address, size_t size) const;

    void InvalidateConfig(gna_request_cfg_id configId,
            LayerConfiguration *layerConfiguration, uint32_t layerIndex) const;

    bool IsPartOfModel(Memory *memory) const;

    status_t Score(
        RequestConfiguration& config,
        RequestProfiler *profiler,
        KernelBuffers *buffers);

    void ValidateBuffer(std::vector<Memory *> &configMemoryList, Memory *memory) const;

    const uint32_t LayerCount;
    const uint32_t GmmCount;

protected:
    std::unique_ptr<HardwareModelScorable> hardwareModel;

private:
    void createSubmodels(const HardwareCapabilities& hwCaps);


    uint32_t scoreAllSubModels(RequestConfiguration& config,
        RequestProfiler *profiler, KernelBuffers *buffers);

    uint32_t getGmmCount(const gna_model *const userModel) const;

    BaseValidator makeValidator(gna_device_generation deviceGeneration);

    const AccelerationDetector &detector;
    std::vector<std::unique_ptr<Memory>>& memoryList;
    std::vector<Memory *> modelMemoryList;
    SoftwareModel softwareModel;
    std::vector<std::unique_ptr<SubModel>> submodels;

};

}
