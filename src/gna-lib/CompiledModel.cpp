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

#include <functional>

#include "Memory.h"
#include "SubModel.h"

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
    softwareModel{ rawModel, gmmCount, validBoundaries },
    submodels{},
    swFastAccel{ detector.GetFastestAcceleration() },
    swSatAccel{ static_cast<acceleration>(detector.GetFastestAcceleration() & GNA_HW) }
{
    createSubmodels(detector);
    auto isHardwareCompliant = !(1 == submodels.size() && SubmodelType::Software == submodels.at(0)->Type);
    if(!isHardwareCompliant)
    {
        Log->Message("WARNING: None of model layers is compliant with selected hardware GNA device, "
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

const size_t CompiledModel::CalculateModelSize(const size_t userSize, const uint16_t layerCount,
    const uint16_t gmmCountIn)
{
    Expect::InRange(userSize, 64, 256 * 1024 * 1024, GNA_INVALIDMEMSIZE);
    auto internalSize = CalculateInternalModelSize(layerCount, gmmCountIn);
    Expect::InRange(internalSize, 1 * sizeof(XNN_LYR), 256 * 1024 * 1024, GNA_INVALIDMEMSIZE);
    auto totalSize = userSize + internalSize;
    return totalSize;
}

const size_t CompiledModel::CalculateInternalModelSize(const uint16_t layerCount,
    const uint16_t gmmCountIn)
{
    // TODO:INTEGRATION: add detector reference to c-tor and calculate hardware size if applicable
    // for model dumper use fake detector in device
    return HardwareModel::CalculateDescriptorSize(layerCount, gmmCountIn);
}

const size_t CompiledModel::CalculateInternalModelSize(const gna_model * rawModel)
{
    uint16_t gmmLayerCount = 0;
    for (auto ix = uint32_t{0}; ix < rawModel->nLayers; ++ix)
    {
        if (INTEL_GMM == rawModel->pLayers[ix].nLayerKind)
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

uint32_t CompiledModel::GetHardwareOffset(const BaseAddressC& address) const
{
    return hardwareModel->GetOffset(address);
}

void CompiledModel::InvalidateConfig(gna_request_cfg_id configId, LayerConfiguration *layerConfiguration, uint32_t layerIndex) const
{
    if (hardwareModel)
    {
        hardwareModel->InvalidateConfig(configId);
    }

    auto layer = GetLayer(layerIndex);
    layer->UpdateKernelConfigs(*layerConfiguration, validBoundaries);
}

status_t CompiledModel::Score(
    RequestConfiguration& config,
    acceleration accel,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    profilerDTscStart(&profiler->scoring);

    auto swAccel = accel;
    if (GNA_AUTO_FAST == accel || GNA_SW_FAST == accel)
    {
        swAccel = swFastAccel;
    }
    if (GNA_AUTO_SAT == accel || GNA_SW_SAT == accel || GNA_HW == accel)
    {
        swAccel = swSatAccel;
    }

    auto status = GNA_SUCCESS;
    if ((GNA_HW == accel && !hardwareModel)
        || (int32_t) accel > (int32_t) swFastAccel)
    {
        status = GNA_CPUTYPENOTSUPPORTED;
    }
    else if(accel >= GNA_SW_SAT && accel <= GNA_AVX2_FAST)
    {
        Log->Message("Processing request using %s acceleration\n", AccelerationDetector::AccelerationToString(swAccel));
        status = softwareModel.Score(0, LayerCount, swAccel, config, profiler, buffers);
    }
    else for (const auto& submodel : submodels)
    {
        uint32_t layerIndex = submodel->LayerIndex;
        uint32_t layerCount = submodel->GetLayerCount();
        switch (submodel->Type)
        {
        case Software:
            Log->Message("Processing submodel using %s acceleration\n", AccelerationDetector::AccelerationToString(swAccel));
            status = softwareModel.Score(layerIndex, layerCount, swAccel, config, profiler, buffers);
            if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                return status;
            break;
        case Hardware:
            Log->Message("Processing submodel using %s acceleration\n", AccelerationDetector::AccelerationToString(GNA_HW));
            status = hardwareModel->Score(layerIndex, layerCount, config, profiler, buffers, xNN);
            if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                return status;
            break;
        case GMMHardware:
            Log->Message("Processing submodel using %s acceleration (GMM Legacy mode)\n", AccelerationDetector::AccelerationToString(GNA_HW));
            status = hardwareModel->Score(layerIndex, 1, config, profiler, buffers, GMM);
            if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                return status;
            break;
        }
    }
    profilerDTscStop(&profiler->scoring);
    profilerDTscStop(&profiler->total);
    return status;
}

void CompiledModel::createSubmodels(const AccelerationDetector& dispatcher)
{
    auto getSubmodelType = [&dispatcher](intel_layer_kind_t layerKind)
    {
        if (dispatcher.IsLayerSupported(layerKind))
        {
            return SubmodelType::Hardware;
        }
        if (INTEL_GMM == layerKind && dispatcher.HasFeature(LegacyGMM))
        {
            return SubmodelType::GMMHardware;
        }
        return SubmodelType::Software;
    };

    auto &layers = softwareModel.Layers;
    auto layerKind = layers.at(0)->Config.Kind;

    auto submodelCount = 0;
    auto smType = getSubmodelType(layerKind);

    submodels.emplace_back(make_unique<SubModel>(smType, 0));

    for (uint16_t layerIx = 1; layerIx < LayerCount; ++layerIx)
    {
        layerKind = layers.at(layerIx)->Config.Kind;
        smType = getSubmodelType(layerKind);

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
