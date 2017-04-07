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

#include "CompiledModel.h"
#include "SubModel.h"

using namespace GNA;

using std::make_unique;
using std::unique_ptr;
using std::vector;

CompiledModel::CompiledModel(gna_model_id modelId, const gna_model *rawModel, const Memory& memoryIn) :
    modelId{modelId},
    memory{memoryIn},
    userModel{rawModel},
    submodels{},
    layerCount{static_cast<uint16_t>(rawModel->nLayers)}
{};

void CompiledModel::CompileSoftwareModel()
{
    softwareModel = make_unique<SoftwareModel>(userModel);
}

void CompiledModel::CompileHardwareModel(const AccelerationDetector& detector)
{
    hardwareModel = make_unique<HardwareModel>(modelId, *softwareModel, memory, detector);
}

void CompiledModel::CreateSubmodels(const AccelerationDetector& dispatcher)
{
    auto layerType = userModel->pLayers->nLayerKind;
    auto submodelCount = 0;
    auto smType = dispatcher.IsLayerSupported(layerType) ? Hardware : Software;
    submodels.emplace_back(make_unique<SubModel>(smType, 0));

    for(uint16_t layerIx = 1; layerIx < layerCount; ++layerIx)
    {
        layerType = userModel->pLayers[layerIx].nLayerKind;
        smType = dispatcher.IsLayerSupported(layerType) ? Hardware : Software;

        if(smType == submodels[submodelCount]->Type)
        {
            submodels[submodelCount]->AddLayer();
        }
        else
        {
            submodels.emplace_back(make_unique<SubModel>(smType, layerIx));
            submodelCount++;
        }
    }
}

void CompiledModel::ClearSubmodels()
{
    submodels.clear();
}

uint16_t CompiledModel::GetLayerCount() const
{
    return layerCount;
}

SoftwareModel& CompiledModel::GetSoftwareModel() const
{
    return *softwareModel.get();
}

HardwareModel& CompiledModel:: GetHardwareModel() const
{
    return *hardwareModel.get();
}

gna_model_id CompiledModel::GetModelId() const
{
    return modelId;
}