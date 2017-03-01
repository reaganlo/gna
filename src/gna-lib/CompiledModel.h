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

#include "HardwareModel.h"
#include "SoftwareModel.h"
#include "SubModel.h"
#include "RequestConfiguration.h"
#include "AccelerationDetector.h"

#include <vector>

namespace GNA 
{

class CompiledModel
{
public:
    CompiledModel(gna_model_id modelId, const gna_model *rawModel)
        : modelId(modelId), userModel(rawModel)
    {
        layerCount = rawModel->nLayers;
    }

    uint16_t GetLayerCount() const;

    HardwareModel& GetHardwareModel() const;

    SoftwareModel& GetSoftwareModel() const;

    void CompileSoftwareModel();

    void CompileHardwareModel();

    void CreateSubmodels(AccelerationDetector& detector);

    void ClearSubmodels();

    decltype(auto) CompiledModel::GetSubmodels() const 
    {
        return (submodels);
    }

private:
    gna_model_id modelId;

    uint16_t layerCount;
    uint16_t gmmCount = 0;

    uint32_t bufferSize = 0;
    const gna_model *userModel;

    std::unique_ptr<HardwareModel> hwModel;
    std::unique_ptr<SoftwareModel> swModel;

    std::vector<std::unique_ptr<SubModel>> submodels;
    std::vector<std::unique_ptr<RequestConfiguration>> configurations;

    CompiledModel(const CompiledModel &) = delete;
    CompiledModel& operator=(const CompiledModel&) = delete;
};

}