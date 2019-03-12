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

#include "AccelerationDetector.h"
#include "ActiveList.h"
#include "Layer.h"
#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "RecurrentLayer.h"
#include "GmmLayer.h"
#include "LayerConfiguration.h"
#include "Macros.h"
#include "Memory.h"
#include "RecurrentLayer.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "Expect.h"

using std::make_unique;

using namespace GNA;

SoftwareModel::SoftwareModel(const gna_model *const network,
    BaseValidator validator, AccelerationMode fastestAccelIn) :
    layerCount{ network->nLayers },
    fastestAccel{ fastestAccelIn }
{
#ifndef NO_ERRCHECK
    Expect::InRange(network->nGroup, ui32_1, XNN_N_GROUP_MAX, XNN_ERR_LYR_CFG);
    Expect::InRange(layerCount, ui32_1,
        HardwareCapabilities::GetMaximumLayerCount(GNA_DEFAULT_VERSION), XNN_ERR_NET_LYR_NO);
    Expect::NotNull(network->pLayers);
#endif
    build(network->pLayers, validator);
}

uint32_t SoftwareModel::Score(
    uint32_t layerIndex,
    uint32_t layerCountIn,
    RequestConfiguration const &requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *fvBuffers,
    GnaOperationMode operationMode)
{
    UNREFERENCED_PARAMETER(profiler);
    UNREFERENCED_PARAMETER(operationMode);

    validateConfiguration(requestConfiguration);

    auto accel = getEffectiveAccelerationMode(requestConfiguration.Acceleration);
    if (NUM_GNA_ACCEL_MODES == accel)
    {
        return GNA_CPUTYPENOTSUPPORTED;
    }
    LogAcceleration(accel);

    auto saturationCount = uint32_t{ 0 };   // scoring saturation counter
    auto executionConfig = ExecutionConfig{const_cast<KernelBuffers const *>(fvBuffers),
        &saturationCount, requestConfiguration.BufferElementCount};

    auto iter = Layers.begin() + layerIndex;
    auto end = iter + layerCountIn;
    for (; iter < end; ++iter)
    {
        const auto& layer = *iter;
        auto found = requestConfiguration.LayerConfigurations.find(layerIndex);
        if (found == requestConfiguration.LayerConfigurations.end())
        {
            // TODO:3:simplify to single Compute as in Cnn2D
            layer->ComputeHidden(accel, executionConfig);
        }
        else
        {
            auto layerConfiguration = found->second.get();
            layer->Compute(*layerConfiguration, accel, executionConfig);
        }

        ++layerIndex;
    }

    return saturationCount;
}

void SoftwareModel::build(const nn_layer* layers, const BaseValidator& validator)
{
    for (auto i = uint32_t{0}; i < layerCount; i++)
    {
        try
        {
            auto layer = layers + i;
            Layers.push_back(Layer::Create(layer, validator));
        }
        catch (const GnaException& e)
        {
            throw GnaModelException(e, i);
        }
        catch (...)
        {
            throw GnaModelException(GnaException(XNN_ERR_LYR_CFG), i);
        }
    }
}

void SoftwareModel::validateConfiguration(const RequestConfiguration& configuration) const
{
    UNREFERENCED_PARAMETER(configuration);
    //TODO:3:review and remove
    //Expect::True(inputLayerCount == configuration.InputBuffersCount, XNN_ERR_NETWORK_INPUTS);
    //Expect::True(outputLayerCount == configuration.OutputBuffersCount, XNN_ERR_NETWORK_OUTPUTS);*/
}

AccelerationMode SoftwareModel::getEffectiveAccelerationMode(AccelerationMode requiredAccel) const
{
    switch (requiredAccel)
    {
        case GNA_AUTO_FAST:
        case GNA_SW_FAST:
            return fastestAccel;
        case GNA_AUTO_SAT:
        case GNA_SW_SAT:
        case GNA_HW:
            return static_cast<AccelerationMode>(fastestAccel & GNA_HW);
            // enforced sw modes
        default:
            if ((int32_t) requiredAccel > (int32_t) fastestAccel)
            {
                return NUM_GNA_ACCEL_MODES;
            }
            else
            {
                return requiredAccel;
            }
    }
}


