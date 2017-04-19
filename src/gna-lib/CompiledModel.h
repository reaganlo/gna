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
    CompiledModel(gna_model_id modelId, const gna_model *rawModel, const Memory& memory);
    ~CompiledModel() = default;
    CompiledModel(const CompiledModel &) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;

    // TODO: most of these methods are here due to invalid object design, need to refactor to get rid of

    uint32_t GetGmmCount() const;
    uint32_t GetHardwareOffset(const BaseAddressC& address) const;
    const std::vector<std::unique_ptr<Layer>>& GetLayers() const;
    decltype(auto) CompiledModel::GetSubmodels() const
    {
        return (submodels);
    }

    void WriteHardwareLayerInputBuffer(const uint32_t layerIndex, PGNA_BUFFER_DESCR &lyrsCfg,
        const ConfigurationBuffer * const buffer) const;
    void WriteHardwareLayerOutputBuffer(const uint32_t layerIndex, PGNA_BUFFER_DESCR &lyrsCfg,
        const ConfigurationBuffer * const buffer) const;
    void WriteHardwareLayerActiveList(const uint32_t layerIndex, HardwareActiveListDescriptor & descriptor) const;

    void CompileSoftwareModel();
    void CompileHardwareModel(const AccelerationDetector& detector);
    void CreateSubmodels(const AccelerationDetector& detector);

    void ValidateConfiguration(const RequestConfiguration& configuration) const;

    const gna_model_id Id;
    const uint16_t LayerCount;
    const gna_model* const UserModel;

private:
    const Memory& memory;
    uint16_t gmmCount = 0;
    uint32_t bufferSize = 0;

    std::unique_ptr<HardwareModel> hardwareModel;
    std::unique_ptr<SoftwareModel> softwareModel;
    std::vector<std::unique_ptr<SubModel>> submodels;
};

}
