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

#include "CompiledModel.h"

#include "Memory.h"
#include "RequestConfiguration.h"
#include "SubModel.h"

#include <functional>

using namespace GNA;

using std::make_unique;
using std::unique_ptr;
using std::vector;

CompiledModel::CompiledModel(gna_model_id modelId, const gna_model *rawModel, Memory& memoryIn, IoctlSender &sender, const AccelerationDetector& detector) :
    Id{ modelId },
    LayerCount{ static_cast<uint16_t>(rawModel->nLayers) },
    memory{ memoryIn },
    ioctlSender{ sender },
    validBoundaries{ [&memoryIn](const void *buffer, const size_t bufferSize)
        { Expect::ValidBoundaries(buffer, bufferSize, memoryIn.GetUserBuffer(), memoryIn.ModelSize); } },
    validator{ GNA_3_0, &validBoundaries }, // TODO:3: Pass actual device/device list
    softwareModel{ rawModel, gmmCount, validator, detector.GetFastestAcceleration() },
    submodels{}
{
    createSubmodels(detector);
    auto isHardwareCompliant = !(1 == submodels.size() && SubmodelType::Software == submodels.at(0)->Type);
    if(!isHardwareCompliant)
    {
        Log->Warning("None of model layers is compliant with selected hardware GNA device, "
            "only software processing is available for this model.\n");
    }
    if (detector.IsHardwarePresent() && isHardwareCompliant)
    {
        auto memoryId = memoryIn.GetId();
        hardwareModel = make_unique<HardwareModel>(Id, softwareModel.Layers, gmmCount, memoryId,
            memoryIn, memoryIn.GetDescriptorsBase(modelId), sender, detector);
        hardwareModel->Build();
    }
}

const size_t CompiledModel::MaximumInternalModelSize = CalculateInternalModelSize(XNN_LAYERS_MAX_COUNT, GMM_LAYERS_MAX_COUNT);

size_t CompiledModel::CalculateModelSize(const size_t userSize, const uint16_t layerCount,
    const uint16_t gmmCountIn)
{
    Expect::InRange<size_t>(userSize, 64, 256 * 1024 * 1024, GNA_INVALIDMEMSIZE);
    auto internalSize = CalculateInternalModelSize(layerCount, gmmCountIn);
    Expect::InRange<size_t>(internalSize,
        1 * LayerDescriptor::GetSize(), 256 * 1024 * 1024, GNA_INVALIDMEMSIZE);
    auto totalSize = userSize + internalSize;
    return totalSize;
}

size_t CompiledModel::CalculateInternalModelSize(const uint16_t layerCount,
    const uint16_t gmmCountIn)
{
    // TODO:INTEGRATION: add detector reference to c-tor and calculate hardware size if applicable
    // for model dumper use fake detector in device
    return HardwareModel::CalculateDescriptorSize(layerCount, gmmCountIn);
}

size_t CompiledModel::CalculateInternalModelSize(const gna_model * rawModel)
{
    uint16_t gmmLayerCount = 0;
    for (auto ix = uint32_t{0}; ix < rawModel->nLayers; ++ix)
    {
        if (INTEL_GMM == rawModel->pLayers[ix].operation)
            gmmLayerCount++;
    }
    return HardwareModel::CalculateDescriptorSize(rawModel->nLayers, gmmLayerCount);
}

uint16_t CompiledModel::GetGmmCount() const
{
    return gmmCount;
}

const std::vector<std::unique_ptr<Layer>>& CompiledModel::GetLayers() const
{
    return softwareModel.Layers;
}

const Layer* CompiledModel::GetLayer(uint32_t layerIndex) const
{
    try
    {
        return softwareModel.Layers.at(layerIndex).get();
    }
    catch (const std::exception&)
    {
        throw GnaException(XNN_ERR_NET_LYR_NO);
    }
}

void CompiledModel::InvalidateConfig(gna_request_cfg_id configId, LayerConfiguration *layerConfiguration, uint32_t layerIndex) const
{
    if (hardwareModel)
    {
        hardwareModel->InvalidateConfig(configId);
    }

    auto layer = GetLayer(layerIndex);
    layer->UpdateKernelConfigs(*layerConfiguration);
}

status_t CompiledModel::Score(
    RequestConfiguration& config,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    profilerDTscStart(&profiler->scoring);

    auto saturationCount = uint32_t{0};
    auto isHardwareEnforced = GNA_HW == config.Acceleration && !hardwareModel;
    auto isSoftwareEnforced = config.Acceleration >= GNA_SW_SAT && config.Acceleration <= GNA_AVX2_FAST;

    try
    {
        if (isHardwareEnforced)
        {
            return GNA_CPUTYPENOTSUPPORTED;
        }
        else if (isSoftwareEnforced)
        {
            saturationCount = softwareModel.Score(0, LayerCount, config, profiler, buffers);
        }
        else
        {
            saturationCount = scoreAllSubModels(config, profiler, buffers);
        }
    }
    catch (const GnaException& e)
    {
        return e.getStatus();
    }
    profilerDTscStop(&profiler->scoring);
    profilerDTscStop(&profiler->total);

    return (saturationCount > 0) ? GNA_SSATURATE : GNA_SUCCESS;
}

uint32_t CompiledModel::scoreAllSubModels(RequestConfiguration& config,
    RequestProfiler *profiler, KernelBuffers *buffers)
{
    auto saturationCount = uint32_t{0};
    for (const auto& submodel : submodels)
    {
        uint32_t layerIndex = submodel->LayerIndex;
        uint32_t layerCount = submodel->GetLayerCount();
        switch (submodel->Type)
        {
        case Software:
        saturationCount += softwareModel.Score(layerIndex, layerCount, config, profiler, buffers);
        break;
        case Hardware:
        saturationCount += hardwareModel->Score(layerIndex, layerCount, config, profiler, buffers, xNN);
        break;
        case GMMHardware:
        saturationCount += hardwareModel->Score(layerIndex, 1, config, profiler, buffers, GMM);
        break;
        }
    }
    return saturationCount;
}

void CompiledModel::createSubmodels(const AccelerationDetector& dispatcher)
{
    auto getSubmodelType = [&dispatcher](nn_operation operation)
    {
        if (dispatcher.IsLayerSupported(operation))
        {
            return SubmodelType::Hardware;
        }
        if (INTEL_GMM == operation && dispatcher.HasFeature(LegacyGMM))
        {
            return SubmodelType::GMMHardware;
        }
        return SubmodelType::Software;
    };

    auto &layers = softwareModel.Layers;
    auto operation = layers.at(0)->Operation;

    auto submodelCount = 0;
    auto smType = getSubmodelType(operation);

    submodels.emplace_back(make_unique<SubModel>(smType, 0));

    for (uint16_t layerIx = 1; layerIx < LayerCount; ++layerIx)
    {
        operation = layers.at(layerIx)->Operation;
        smType = getSubmodelType(operation);

        if (GMMHardware == smType || submodels.at(submodelCount)->Type != smType)
        {
            submodels.emplace_back(make_unique<SubModel>(smType, layerIx));
            submodelCount++;
        }
        else
        {
            // exceeded supported number of layers
            if (Hardware == smType && !dispatcher.HasFeature(Layer8K)
                && XNN_LAYERS_MAX_COUNT_OLD <= submodels.at(submodelCount)->GetLayerCount())
            {
                submodels.emplace_back(make_unique<SubModel>(smType, layerIx));
                submodelCount++;
            }
            else submodels[submodelCount]->AddLayer();
        }
    }
}
