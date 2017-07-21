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

#include <functional>

#include "Memory.h"
#include "SubModel.h"

using namespace GNA;

using std::make_unique;
using std::unique_ptr;
using std::vector;

CompiledModel::CompiledModel(gna_model_id modelId, const gna_model *rawModel, Memory& memoryIn, const AccelerationDetector& detector) :
    Id{ modelId },
    LayerCount{ static_cast<uint16_t>(rawModel->nLayers) },
    UserModel{ rawModel },
    validBoundaries{ [&memoryIn](const void *buffer, const size_t bufferSize)
        { Expect::ValidBoundaries(buffer, bufferSize, memoryIn.GetUserBuffer(), memoryIn.ModelSize); } },
    softwareModel{ UserModel, gmmCount, validBoundaries },
    submodels{},
    swFastAccel{ detector.GetFastestAcceleration() },
    swSatAccel{ static_cast<acceleration>(detector.GetFastestAcceleration() & GNA_HW) }
{
    if (detector.IsHardwarePresent())
    {
        hardwareModel = make_unique<HardwareModel>(Id, softwareModel.Layers, gmmCount, memoryIn, detector);
    }

    createSubmodels(detector);
    prepareScoreMethods(detector);
};

const size_t CompiledModel::MaximumInternalModelSize = CalculateInternalModelSize(XNN_LAYERS_MAX_COUNT, GMM_LAYERS_MAX_COUNT);

const size_t CompiledModel::CalculateModelSize(const size_t userSize, const uint16_t layerCount,
    const uint16_t gmmCountIn)
{
    auto internalSize = CalculateInternalModelSize(layerCount, gmmCountIn);
    auto totalSize = userSize + internalSize;
    Expect::InRange(totalSize, 1, 256 * 1024 * 1024, GNA_INVALIDMEMSIZE);
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
    for (auto ix = 0ui32; ix < rawModel->nLayers; ++ix)
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

uint32_t CompiledModel::GetHardwareOffset(const BaseAddressC& address) const
{
    return hardwareModel->GetOffset(address);
}

void CompiledModel::InvalidateConfig(gna_request_cfg_id configId, LayerConfiguration *layerConfiguration, uint32_t layerIndex) const
{
    if (hardwareModel)
    {
        hardwareModel->InvalidateConfigCache(configId);
    }

    auto layer = GetLayers().at(layerIndex).get();
    layer->UpdateKernelConfigs(*layerConfiguration, validBoundaries);
}

status_t CompiledModel::Score(
    RequestConfiguration& config,
    acceleration accel,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    profilerDTscAStart(&profiler->scoring);

    auto swAccel = accel;
    if (GNA_AUTO_FAST == accel || GNA_SW_FAST == accel || GNA_HW == accel)
    {
        swAccel = swFastAccel;
    }
    if (GNA_AUTO_SAT == accel || GNA_SW_SAT == accel)
    {
        swAccel = swSatAccel;
    }

    auto status = GNA_SUCCESS;
    auto scoreMethod = scoreMethods[accel];
    switch (scoreMethod)
    {
    case SoftwareOnly:
        status = softwareModel.Score(0, LayerCount, swAccel, config, profiler, buffers);
        break;
    case HardwareOnly:
        status = hardwareModel->Score(0, LayerCount, config, profiler, buffers);
        break;
    case Mixed:
    {
        for (const auto& submodel : submodels)
        {
            uint32_t layerIndex = submodel->LayerIndex;
            uint32_t layerCount = submodel->GetLayerCount();
            switch (submodel->Type)
            {
            case Software:
                status = softwareModel.Score(layerIndex, layerCount, swAccel, config, profiler, buffers);
                if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                    return status;
                break;
            case Hardware:
                status = hardwareModel->Score(layerIndex, layerCount, config, profiler, buffers);
                if (status != GNA_SUCCESS && status != GNA_SSATURATE)
                    return status;
                break;
            case GMMHardware:
                throw GnaException(GNA_CPUTYPENOTSUPPORTED);
            }
        }
        break;
    }
    }
    profilerDTscStop(&profiler->scoring);
    profilerDTscStop(&profiler->total);
    return status;
}

void CompiledModel::createSubmodels(const AccelerationDetector& dispatcher)
{
    auto layerType = UserModel->pLayers->nLayerKind;
    auto submodelCount = 0;
    auto smType = dispatcher.IsLayerSupported(layerType) ? Hardware : Software;
    submodels.emplace_back(make_unique<SubModel>(smType, 0));

    for (uint16_t layerIx = 1; layerIx < LayerCount; ++layerIx)
    {
        layerType = UserModel->pLayers[layerIx].nLayerKind;
        smType = dispatcher.IsLayerSupported(layerType) ? Hardware : Software;

        if (smType == submodels[submodelCount]->Type)
        {
            // exceeded supported number of layers
            if (Hardware == smType && !dispatcher.HasFeature(Layer8K)
                && XNN_LAYERS_MAX_COUNT_OLD == submodels[submodelCount]->GetLayerCount())
            {
                submodels.emplace_back(make_unique<SubModel>(smType, layerIx));
                submodelCount++;
            }
            else submodels[submodelCount]->AddLayer();
        }
        else
        {
            submodels.emplace_back(make_unique<SubModel>(smType, layerIx));
            submodelCount++;
        }
    }
}

void CompiledModel::prepareScoreMethods(const AccelerationDetector& detector)
{
    scoreMethods[GNA_GEN_FAST] = SoftwareOnly;
    scoreMethods[GNA_GEN_SAT] = SoftwareOnly;

    scoreMethods[GNA_SSE4_2_FAST] = SoftwareOnly;
    scoreMethods[GNA_SSE4_2_SAT] = SoftwareOnly;

    scoreMethods[GNA_AVX1_FAST] = SoftwareOnly;
    scoreMethods[GNA_AVX1_SAT] = SoftwareOnly;

    scoreMethods[GNA_AVX2_FAST] = SoftwareOnly;
    scoreMethods[GNA_AVX2_SAT] = SoftwareOnly;

    scoreMethods[GNA_SW_FAST] = SoftwareOnly;
    scoreMethods[GNA_SW_SAT] = SoftwareOnly;

    if (!detector.IsHardwarePresent())
    {
        scoreMethods[GNA_HW] = None;
        scoreMethods[GNA_AUTO_FAST] = SoftwareOnly;
        scoreMethods[GNA_AUTO_SAT] = SoftwareOnly;
    }
    else
    {
        if (submodels.size() > 1)
        {
            scoreMethods[GNA_HW] = Mixed;
            scoreMethods[GNA_AUTO_FAST] = Mixed;
            scoreMethods[GNA_AUTO_SAT] = Mixed;
        }
        else
        {
            switch (submodels.front()->Type)
            {
                // hardware does not support model
            case Software:
                scoreMethods[GNA_HW] = SoftwareOnly;
                scoreMethods[GNA_AUTO_FAST] = SoftwareOnly;
                scoreMethods[GNA_AUTO_SAT] = SoftwareOnly;
                break;
                // hardware can handle whole model
            case Hardware:
                scoreMethods[GNA_HW] = HardwareOnly;
                scoreMethods[GNA_AUTO_FAST] = HardwareOnly;
                scoreMethods[GNA_AUTO_SAT] = HardwareOnly;
                break;
            default:
                scoreMethods[GNA_HW] = None;
                scoreMethods[GNA_AUTO_FAST] = None;
                scoreMethods[GNA_AUTO_SAT] = None;
                break;
            }
        }
    }
}
