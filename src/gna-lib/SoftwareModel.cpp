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
#include "Memory.h"
#include "RecurrentLayer.h"
#include "Request.h"
#include "RequestConfiguration.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

SoftwareModel::SoftwareModel(const gna_model *const network, ValidBoundariesFunctor validBoundaries) :
    layerCount{ network->nLayers },
    inputVectorCount{ network->nGroup }
{
#ifndef NO_ERRCHECK
    Expect::InRange(inputVectorCount, 1, XNN_N_GROUP_MAX, XNN_ERR_LYR_CFG);
    Expect::InRange(layerCount, 1, XNN_LAYERS_MAX_COUNT, XNN_ERR_NET_LYR_NO);
    Expect::NotNull(network->pLayers);
#endif
    build(network->pLayers);
    validate(validBoundaries);
}

status_t SoftwareModel::Score(
    uint32_t layerIndex,
    uint32_t layerCountIn,
    acceleration accel,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *fvBuffers)
{
    UNREFERENCED_PARAMETER(profiler);
    validateConfiguration(requestConfiguration);

    const uint32_t* activeIndices = nullptr; // active list pointer
    auto saturationCount = uint32_t{ 0 };   // scoring saturation counter

    auto iter = Layers.begin() + layerIndex;
    auto end = iter + layerCountIn;
    for (; iter < end; ++iter)
    {
        const auto& layer = *iter;
        if (INTEL_HIDDEN == layer->Config.Type)
        {
            layer->ComputeHidden(accel, fvBuffers, &saturationCount);
        }
        else 
        {
            auto found = requestConfiguration.LayerConfigurations.find(layerIndex);
            if (found != requestConfiguration.LayerConfigurations.end())
            {
                auto layerConfiguration = found->second.get();
                layer->ComputeConfig(*layerConfiguration, accel, fvBuffers, &saturationCount);
            }
            else
            {
                throw GnaException{ XNN_ERR_LYR_CFG };
            }
        }

        ++layerIndex;
    }

    return (saturationCount > 0) ? GNA_SSATURATE : GNA_SUCCESS;
}

void SoftwareModel::validateConfiguration(const RequestConfiguration& configuration) const
{
    Expect::True(inputLayerCount == configuration.InputBuffersCount, XNN_ERR_NETWORK_INPUTS);
    Expect::True(outputLayerCount == configuration.OutputBuffersCount, XNN_ERR_NETWORK_OUTPUTS);
}

void SoftwareModel::build(const nn_layer* layers)
{
    for (auto i = 0ui32; i < layerCount; i++)
    {
        auto layer = layers + i;
        Layers.push_back(Layer::Create(const_cast<const nn_layer*>(layer)));
        switch (layer->type)
        {
        case INTEL_INPUT:
            ++inputLayerCount;
            break;
        case INTEL_OUTPUT:
            ++outputLayerCount;
            break;
        case INTEL_INPUT_OUTPUT:
            ++inputLayerCount;
            ++outputLayerCount;
            break;
        }
    }
}

void SoftwareModel::validate(ValidBoundariesFunctor validBoundaries) const
{
    for (const auto& layer : Layers)
    {
        if (layer->Config.Type != INTEL_INPUT
            && layer->Config.Type != INTEL_INPUT_OUTPUT)
        {
            validBoundaries(layer->Input.Buffer.Get(), layer->Input.BufferSize);
        }
        if (layer->Config.Type != INTEL_OUTPUT
            && layer->Config.Type != INTEL_INPUT_OUTPUT)
        {
            validBoundaries(layer->Output.Buffer.Get(), layer->Output.BufferSize);
        }

        const auto layerKind = layer->Config.Kind;
        const auto affineLayer = dynamic_cast<AffineBaseLayer*>(layer.get());
        const ActivationFunction *activation = nullptr;

        if (affineLayer)
        {
            activation = affineLayer->Activation.get();

            size_t biasesSize = layer->Output.ElementCount;
            size_t weightsSize = (GNA_WEIGHT_2B == affineLayer->Affine->Mode ? sizeof(int16_t) : sizeof(int8_t));

            if (INTEL_AFFINE_MULTIBIAS == layerKind)
            {
                biasesSize *= sizeof(nn_bias_s);
                if (GNA_WEIGHT_1B == affineLayer->Affine->Mode)
                {
                    size_t weightScaleSize = layer->Output.ElementCount * sizeof(nn_bias_c);
                    auto functionMulti = static_cast<const AffineFunctionMulti1B*>(affineLayer->Affine.get());
                    validBoundaries(functionMulti->WeightScaleFactors, weightScaleSize);
                }
            }
            else
            {
                biasesSize *= (GNA_WEIGHT_2B == affineLayer->Affine->Mode ? sizeof(nn_bias_s) : sizeof(nn_bias_c));
            }

            if (INTEL_RECURRENT == layerKind)
            {
                weightsSize *= ((layer->Output.ElementCount + layer->Input.ElementCount) * layer->Output.ElementCount);

                if (INTEL_OUTPUT != layer->Config.Type && INTEL_INPUT_OUTPUT != layer->Config.Type)
                {
                    auto rnnLayer = static_cast<RnnLayer*>(layer.get());
                    auto feedbackBuffer = layer->Output.Buffer.Get() - rnnLayer->FeedbackDelay * layer->Output.ElementCount;
                    auto feedbackSize = layer->Output.ElementCount;
                    validBoundaries(feedbackBuffer, feedbackSize);
                }
            }
            else if (INTEL_AFFINE_DIAGONAL == layerKind)
            {
                weightsSize *= layer->Output.ElementCount;
            }
            else /* INTEL_AFFINE || INTEL_AFFINE_MULTIBIAS */
            {
                weightsSize *= layer->Input.ElementCount * layer->Output.ElementCount;
            }

            validBoundaries(affineLayer->Affine->Biases, biasesSize);
            validBoundaries(affineLayer->Affine->Weights, weightsSize);
        }
        else if (INTEL_CONVOLUTIONAL == layerKind)
        {
            auto cnnLayer = static_cast<CnnLayer*>(layer.get());
            activation = cnnLayer->Activation.get();

            size_t filtersSize = cnnLayer->Convolution.Filters.Count * cnnLayer->Convolution.Filters.CoefficientCount;
            validBoundaries(cnnLayer->Convolution.Filters.Data, filtersSize);

            break;
        }
        else if (INTEL_GMM == layerKind)
        {
            auto gmmLayer = static_cast<GmmLayer*>(layer.get());
            validBoundaries(gmmLayer->Data.gaussianConstants, gmmLayer->Params.GaussConstSetOffsetSize);
            validBoundaries(gmmLayer->Data.meanValues, gmmLayer->Params.MeanSetOffsetSize);
            if (GNA_MAXMIX16 == gmmLayer->Config.mode)
            {
                validBoundaries(gmmLayer->Data.inverseCovariancesForMaxMix16, gmmLayer->Params.VarianceSize);
            }
            else
            {
                validBoundaries(gmmLayer->Data.inverseCovariancesForMaxMix8, gmmLayer->Params.VarianceSize);
            }
            break;
        }

        if (activation)
        {
            auto scratchpadSize = layer->Output.ElementCount * layer->Output.VectorCount * LayerOutput::ActivatedOutputSize;
            validBoundaries(layer->Output.ScratchPad, scratchpadSize);

            auto segmentsSize = activation->SegmentCount * sizeof(nn_pwl_seg);
            validBoundaries(activation->Segments, segmentsSize);
        }
    }
}
