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

#include "SoftwareModel.h"

#include "Expect.h"
#include "HardwareCapabilities.h"
#include "KernelArguments.h"
#include "Layer.h"
#include "Macros.h"
#include "RequestConfiguration.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"

#include <cstdint>
#include <functional>
#include <map>
#include <utility>

using namespace GNA;

void SoftwareModel::CheckModel(uint32_t declaredBatchSize, void * operationPointer) const
{
    Expect::InRange(declaredBatchSize, ui32_1, XNN_N_GROUP_MAX,
        Gna2StatusXnnErrorLyrCfg);
    Expect::InRange(layerCount, ui32_1,
        HardwareCapabilities::GetMaximumLayerCount(DefaultDeviceVersion),
        Gna2StatusXnnErrorNetLyrNo);
    Expect::NotNull(operationPointer);
}
//TODO:3:P2: change to template
SoftwareModel::SoftwareModel(const gna_model& network,
    BaseValidator validator,
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn) :
    layerCount{ network.nLayers },
    supportedCpuAccelerations { supportedCpuAccelerationsIn }
{
    CheckModel(network.nGroup, network.pLayers);
    build(network.pLayers, validator);
}

SoftwareModel::SoftwareModel(const Gna2Model& model,
    BaseValidator validator,
    const std::vector<Gna2AccelerationMode>& supportedCpuAccelerationsIn) :
    layerCount{ model.NumberOfOperations },
    supportedCpuAccelerations{ supportedCpuAccelerationsIn }
{
    Expect::NotNull(model.Operations);
    build(model.Operations, validator);
}

uint32_t SoftwareModel::Score(
    uint32_t layerIndex,
    uint32_t layerCountIn,
    RequestConfiguration const &requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *fvBuffers)
{
    UNREFERENCED_PARAMETER(profiler);

    validateConfiguration(requestConfiguration);

    const auto accel = requestConfiguration.Acceleration.GetEffectiveSoftwareAccelerationMode(supportedCpuAccelerations);

    LogAcceleration(accel);

    auto config = InferenceConfig{fvBuffers, requestConfiguration};
    auto iter = Layers.begin() + layerIndex;
    auto end = iter + layerCountIn;
    for (; iter < end; ++iter)
    {
        auto& layer = *iter;
        auto found = requestConfiguration.LayerConfigurations.find(layerIndex);
        if (found == requestConfiguration.LayerConfigurations.end())
        {
            // TODO:3:simplify to single Compute as in Cnn2D
            layer->ComputeHidden(accel, config.GetEffective(*layer));
        }
        else
        {
            auto layerConfiguration = found->second.get();
            layer->Compute(*layerConfiguration, accel, config.GetEffective(*layer));
        }

        ++layerIndex;
    }

    return config.SaturationCount;
}

void SoftwareModel::validateConfiguration(const RequestConfiguration& configuration) const
{
    UNREFERENCED_PARAMETER(configuration);
    //TODO:3:review and remove
    //Expect::True(inputLayerCount == configuration.InputBuffersCount, Gna2StatusXnnErrorNetworkInputs);
    //Expect::True(outputLayerCount == configuration.OutputBuffersCount, Gna2StatusXnnErrorNetworkOutputs);*/
}

InferenceConfig::InferenceConfig(KernelBuffers* fvBuffers,
    RequestConfiguration const& requestConfiguration) :
    SaturationCount{0}
{
    auto buffers = const_cast<KernelBuffers const *>(fvBuffers);
    executionConfig = std::make_unique<ExecutionConfig>(buffers,
        &SaturationCount, requestConfiguration.BufferElementCount);
    auto const isAdl = HardwareCapabilities::IsAdlDevice(requestConfiguration.GetConsistentDevice());
    hasAdlConsistency = isAdl && requestConfiguration.Acceleration.GetHwConsistency();
    if (hasAdlConsistency)
    {
        executionConfigAdl = std::make_unique<ExecutionConfig>(buffers,
            &SaturationCount, requestConfiguration.BufferElementCountForAdl);
        getEffective = &InferenceConfig::getForAdlFix;
    }
    else
    {
        getEffective = &InferenceConfig::getNormal;
    }
}

ExecutionConfig& InferenceConfig::getForAdlFix(Layer& layer) const
{
    if (layer.VerifyHas1BInputAnd2BWeight())
    {
        return *executionConfigAdl;
    }
    return *executionConfig;
}

ExecutionConfig& InferenceConfig::getNormal(Layer& layer) const
{
    UNREFERENCED_PARAMETER(layer);
    return *executionConfig;
}
