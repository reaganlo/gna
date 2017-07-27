/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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
#include "HardwareModel.h"
#include "SoftwareModel.h"
#include "SubModel.h"

namespace GNA
{

struct ConfigurationBuffer;

class CompiledModel
{
public:
    CompiledModel(gna_model_id modelId, const gna_model *rawModel, Memory& memory, IoctlSender &sender, const AccelerationDetector& detector);
    virtual ~CompiledModel() = default;
    CompiledModel(const CompiledModel &) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;

    // TODO: most of these methods are here due to invalid object design, need to refactor to get rid of
    static const size_t CalculateModelSize(const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount);
    static const size_t CalculateInternalModelSize(const uint16_t layerCount, const uint16_t gmmCount);
    static const size_t CalculateInternalModelSize(const gna_model * rawModel);

    uint16_t GetGmmCount() const;
    uint32_t GetHardwareOffset(const BaseAddressC& address) const;
    const std::vector<std::unique_ptr<Layer>>& GetLayers() const;
    decltype(auto) CompiledModel::GetSubmodels() const
    {
        return (submodels);
    }

    void InvalidateConfig(gna_request_cfg_id configId, LayerConfiguration *layerConfiguration, uint32_t layerIndex) const;

    status_t Score(
        RequestConfiguration& config,
        acceleration accel,
        RequestProfiler *profiler,
        KernelBuffers *buffers);

    static const size_t MaximumInternalModelSize;
    const gna_model_id Id;
    const uint16_t LayerCount;
    
protected:
    Memory& memory;
    IoctlSender &ioctlSender;
    ValidBoundariesFunctor validBoundaries;
    uint16_t gmmCount = 0;
    uint32_t bufferSize = 0;

    SoftwareModel softwareModel;
    std::unique_ptr<HardwareModel> hardwareModel;

    std::vector<std::unique_ptr<SubModel>> submodels;

    const acceleration swFastAccel;
    const acceleration swSatAccel;

    void createSubmodels(const AccelerationDetector& detector);
};

}
