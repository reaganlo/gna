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

#include "ConvolutionalLayer.h"

#include "AccelerationDetector.h"
#include "LayerConfiguration.h"
#include "Validator.h"

using std::make_unique;
using std::unique_ptr;

using namespace GNA;

FiltersConfig::FiltersConfig(const nn_layer_conv * sourceLayer, const uint32_t inputElementCount) :
    Count{ sourceLayer->nFilters },
    CoefficientCount{ sourceLayer->nFilterCoefficients },
    Data{ static_cast<int16_t*>(sourceLayer->pFilters) },
    Biases{ sourceLayer->pBiases }
{
    Expect::InRange(sourceLayer->nFilterRows, 1, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
    Expect::True(sourceLayer->nBytesFilterCoefficient == 2, XNN_ERR_WEIGHT_BYTES);

    Expect::InRange(Count, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_ERR_FLT_COUNT);
    Expect::MultiplicityOf(Count, CNN_N_FLT_COEFF_MPLY);

    Expect::InRange(CoefficientCount, CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
    Expect::True(CoefficientCount <= inputElementCount, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(CoefficientCount, XNN_N_IN_ELEMS_MPLY);

    Expect::ValidBuffer(Data);
    Expect::True(sizeof(nn_bias_s) == sourceLayer->nBytesBias, XNN_ERR_BIAS_BYTES);
}

FeatureMaps::FeatureMaps(const nn_layer_conv * sourceLayer) :
    Count{ sourceLayer->nFeatureMaps },
    RowCount{ sourceLayer->nFeatureMapRows },
    ColumnCount{ sourceLayer->nFeatureMapColumns },
    Stride{ Count * ColumnCount } // always move 1 "row"
{
    Expect::InRange(Stride, 1, CNN_N_FLT_COEFF_MAX, CNN_ERR_FLT_STRIDE);
}

ConvolutionFunction::ConvolutionFunction(const nn_layer_conv * sourceLayer, const uint32_t inputElementCount,
    int16_t const * const inputs, int32_t * const outputs) :
    Filters{ sourceLayer, inputElementCount },
    FeatMaps{ sourceLayer },
    OutputElementsCount{ (inputElementCount - Filters.CoefficientCount) / FeatMaps.Stride + 1 },
    kernels{AccelerationDetector::GetKernelMap<ConvolutionKernel>()},
    hiddenConfig{FeatMaps.Stride, OutputElementsCount, Filters.Count, Filters.CoefficientCount,
        inputs, Filters.Data, Filters.Biases, outputs}
{
    Expect::InRange(FeatMaps.Stride, 1, Filters.CoefficientCount, XNN_ERR_LYR_CFG);
    auto featureCount = FeatMaps.RowCount * FeatMaps.Stride;
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, XNN_ERR_LYR_CFG);
}

unique_ptr<const ConvolutionConfig> ConvolutionFunction::GetRunConfig(int16_t const * const inputs, int32_t * const outputs) const
{
    return make_unique<const ConvolutionConfig>(&hiddenConfig, inputs, outputs);
}

void ConvolutionFunction::ComputeHidden(acceleration accel, uint32_t *saturationCount) const
{
    auto convConfig = ConvolutionConfig{&hiddenConfig, saturationCount};

    kernels.at(accel)(&convConfig);
}

void ConvolutionFunction::ComputeConfig(const ConvolutionConfig* const config, acceleration accel, uint32_t *saturationCount) const
{
    auto convConfig = ConvolutionConfig{config, saturationCount};

    kernels.at(accel)(&convConfig);
}

PoolingFunction::PoolingFunction(const nn_layer_conv * sourceLayer) :
    Type{sourceLayer->poolType},
    Size{sourceLayer->nPoolSize},
    Stride{sourceLayer->nPoolStride},
    kernels{AccelerationDetector::GetKernelMap<ConvolutionPoolingKernel>()},
    hiddenConfig{Type, Size, Stride}
{
    Expect::InRange(Type, INTEL_NO_POOLING, NUM_POOLING_TYPES - 1, XNN_ERR_LYR_CFG);
    if (INTEL_NO_POOLING != Type)
    {
        Expect::InRange(Size, CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, XNN_ERR_LYR_CFG);
        Expect::InRange(Stride, CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, CNN_ERR_POOL_STRIDE);
    }
}

void PoolingFunction::Compute(const ConvolutionConfig * convolutionConfig, acceleration accel, int64_t * poolScratchPad,
    const PwlCached * pwl) const
{
    auto poolConfig = PoolingConfig{&hiddenConfig, poolScratchPad};
    kernels.at(accel)(convolutionConfig, &poolConfig, pwl);
}

CnnLayer::CnnLayer(nn_layer const * const layer) :
    Layer(layer),
    Activation(ActivationFunction::Create(layer->nLayerKind, layer->pLayerStruct, Output.ScratchPad,
        PwlOutputConfig{Output.ElementCount * Output.VectorCount, Output.ScratchPad, Output.Buffer})),
    Pooling{static_cast<const nn_layer_conv*>(layer->pLayerStruct)},
    Convolution{static_cast<const nn_layer_conv*>(layer->pLayerStruct), Input.ElementCount, Input.Buffer,
        (INTEL_NO_POOLING != Pooling.Type) ? Output.Buffer.Get<int32_t>() : (Activation ? Output.ScratchPad : Output.Buffer.Get<int32_t>())}
{
    Expect::True(Input.VectorCount == 1, XNN_ERR_GROUPING);
    Expect::True(Input.VectorCount == Output.VectorCount, XNN_ERR_GROUPING);
    Output.SetOutputMode(Activation.operator bool(), layer->nBytesPerOutput);

    if (INTEL_NO_POOLING == Pooling.Type)
    {
        if (Activation)
        {
            Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
            {this->computeHiddenPwl(accel, fvBuffers, saturationCount); };

            Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
            {this->computeConfigPwl(layerConfiguration, accel, fvBuffers, saturationCount); };
        }
        else
        {
            Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
            {this->computeHidden(accel, fvBuffers, saturationCount); };

            Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
            {this->computeConfig(layerConfiguration, accel, fvBuffers, saturationCount); };
        }
    }
    else
    {
        Expect::True(nullptr != Activation, XNN_ERR_PWL_SEGMENTS); // Activation is required for cnn with pooling
        Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeHiddenPool(accel, fvBuffers, saturationCount); };

        Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeConfigPool(layerConfiguration, accel, fvBuffers, saturationCount); };
    }
}

void CnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration, ValidBoundariesFunctor validBoundaries) const
{
    Layer::UpdateKernelConfigs(layerConfiguration, validBoundaries);

    auto inputBuffer = Input.Buffer;
    if (layerConfiguration.InputBuffer)
    {
        inputBuffer = *layerConfiguration.InputBuffer;
        validBoundaries(inputBuffer, Input.BufferSize);
    }

    auto filterOutputBuffer = Activation ? Output.ScratchPad :
        (layerConfiguration.OutputBuffer
         ? layerConfiguration.OutputBuffer->Get<int32_t>() : Output.Buffer.Get<int32_t>());

    auto pwlOutputBuffer = layerConfiguration.OutputBuffer
        ? *layerConfiguration.OutputBuffer
        : Output.Buffer;

    if (layerConfiguration.OutputBuffer)
    {
        auto outputSize = Convolution.Filters.Count * Convolution.OutputElementsCount;
        if (Activation)
        {
            outputSize *= LayerOutput::ActivatedOutputSize;
            validBoundaries(pwlOutputBuffer, outputSize);
        }
        else
        {
            outputSize *= LayerOutput::NonActivatedOutputSize;
            validBoundaries(filterOutputBuffer, outputSize);
        }
    }

    auto& configs = layerConfiguration.Configs;
    if (INTEL_NO_POOLING == Pooling.Type)
    {
        configs.Convolution = Convolution.GetRunConfig(inputBuffer, filterOutputBuffer);
        if (Activation)
        {
            configs.PwlOutput = Activation->GetOutputConfig(pwlOutputBuffer);
        }
    }
    else
    {
        configs.Convolution = Convolution.GetRunConfig(inputBuffer, pwlOutputBuffer);
    }
}

void CnnLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);

    Convolution.ComputeHidden(accel, saturationCount);
}

void CnnLayer::computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);

    Convolution.ComputeHidden(accel, saturationCount);
    Activation->ComputeHidden(accel, saturationCount);
}

void CnnLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);

    Convolution.ComputeConfig(layerConfiguration.Configs.Convolution.get(), accel, saturationCount);
}

void CnnLayer::computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    UNREFERENCED_PARAMETER(fvBuffers);

    Convolution.ComputeConfig(layerConfiguration.Configs.Convolution.get(), accel, saturationCount);
    Activation->ComputeConfig(layerConfiguration, accel, saturationCount);
}

void CnnLayer::computeHiddenPool(acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    auto convConfig = ConvolutionConfig{Convolution.GetHiddenConfig(), saturationCount};
    Pooling.Compute(&convConfig, accel, fvBuffers->pool, &Activation->Pwl);
}

void CnnLayer::computeConfigPool(LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    auto convConfig = ConvolutionConfig{layerConfiguration.Configs.Convolution.get(), saturationCount};
    Pooling.Compute(&convConfig, accel, fvBuffers->pool, &Activation->Pwl);
}
