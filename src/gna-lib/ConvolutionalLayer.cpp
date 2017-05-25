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

using namespace GNA;

FiltersConfig::FiltersConfig(const nn_layer_conv * sourceLayer, const uint32_t inputElementCount) :
    BiasSimple{ sourceLayer->nBytesBias, sourceLayer->pBiases },
    Count{ sourceLayer->nFilters },
    CoefficientCount{ sourceLayer->nFilterCoefficients },
    Data{ static_cast<int16_t*>(sourceLayer->pFilters) }
{
    Expect::InRange(sourceLayer->nFilterRows, 1, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
    Expect::True(sourceLayer->nBytesFilterCoefficient == 2, XNN_ERR_WEIGHT_BYTES);

    Expect::InRange(Count, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_ERR_FLT_COUNT);
    Expect::MultiplicityOf(Count, CNN_N_FLT_COEFF_MPLY);

    Expect::InRange(CoefficientCount, CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
    Expect::True(CoefficientCount <= inputElementCount, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(CoefficientCount, XNN_N_IN_ELEMS_MPLY);

    Expect::ValidBuffer(Data);
}

FeatureMaps::FeatureMaps(const nn_layer_conv * sourceLayer) :
    Count{ sourceLayer->nFeatureMaps },
    RowCount{ sourceLayer->nFeatureMapRows },
    ColumnCount{ sourceLayer->nFeatureMapColumns },
    Stride{ Count * ColumnCount } // always move 1 "row"
{
    Expect::InRange(Stride, 1, CNN_N_FLT_COEFF_MAX, CNN_ERR_FLT_STRIDE);
}

ConvolutionFunction::ConvolutionFunction(const nn_layer_conv * sourceLayer, const uint32_t inputElementCount) :
    Filters{ sourceLayer, inputElementCount },
    FeatureMaps{ sourceLayer },
    OutputElementsCount{ (inputElementCount - Filters.CoefficientCount) / FeatureMaps.Stride + 1 }
{
    auto featureCount = FeatureMaps.RowCount * FeatureMaps.Stride;
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, XNN_ERR_LYR_CFG);
}

PoolingFunction::PoolingFunction(const nn_layer_conv * sourceLayer) :
    Type{ sourceLayer->poolType },
    Size{ sourceLayer->nPoolSize },
    Stride{ sourceLayer->nPoolStride }
{
    Expect::InRange(Type, INTEL_NO_POOLING, NUM_POOLING_TYPES - 1, XNN_ERR_LYR_CFG);
    if (INTEL_NO_POOLING != Type)
    {
        Expect::InRange(Size, CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, XNN_ERR_LYR_CFG);
        Expect::InRange(Stride, CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, CNN_ERR_POOL_STRIDE);
    }
}

CnnLayer::CnnLayer(nn_layer const * const layer) :
    Layer(layer),
    // CNN has only 2B output with Activation always enabled
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_conv*>(layer->pLayerStruct)->pwl, false)),
    Convolution{ static_cast<const nn_layer_conv*>(layer->pLayerStruct), Input.ElementCount },
    Pooling{ static_cast<const nn_layer_conv*>(layer->pLayerStruct) },
    filterKernels{ AccelerationDetector::GetKernelMap<ConvolutionKernel>() },
    poolingKernels{ AccelerationDetector::GetKernelMap<ConvolutionPoolingKernel>() },
    pwlKernels{ AccelerationDetector::GetKernelMap<PwlKernel>() },
    convolutionHiddenConfig{ Convolution.FeatureMaps.Stride, Convolution.OutputElementsCount,
        Convolution.Filters.Count, Convolution.Filters.CoefficientCount, Input.Buffer.Get<int16_t>(), Convolution.Filters.Data,
        Convolution.Filters.Biases, Activation ? reinterpret_cast<int16_t * const>(Output.ScratchPad) : Output.Buffer.Get<int16_t>(), nullptr },
    poolingHiddenConfig{ Pooling.Type, Pooling.Size, Pooling.Stride, nullptr },
    pwlFilterConfig{Output.ScratchPad, Activation ? Activation->Segments : nullptr, Activation ? Activation->SegmentCount : 0},
    pwlPoolConfig{nullptr, Activation ? Activation->Segments : nullptr, Activation ? Activation->SegmentCount : 0},
    pwlOutputConfig{0, Output.ElementCount - 1, 0, Output.VectorCount - 1, Output.VectorCount, nullptr, Output.Buffer}    
{
    Expect::True(Input.VectorCount == 1, XNN_ERR_GROUPING);
    Expect::True(Input.VectorCount == Output.VectorCount, XNN_ERR_GROUPING);

    Output.SetOutputMode(Activation.operator bool(), layer->nBytesPerOutput);

    auto outputElementCount = Convolution.OutputElementsCount; // INTEL_NO_POOLING use convolution outputs per filter
    if (INTEL_NO_POOLING != Pooling.Type) // use pooled outputs per filter
    {
        Expect::True(nullptr != Activation, XNN_ERR_PWL_SEGMENTS); // Activation is required for cnn with pooling
        outputElementCount = ((Convolution.OutputElementsCount - 1) / Pooling.Stride + 1);
    }
    Expect::True(Output.ElementCount == Convolution.Filters.Count * outputElementCount, XNN_ERR_LYR_CFG);
    // NOTE: intentional const override for Output.ElementCount 
    // TODO: consider refactoring
    ((uint32_t)Output.ElementCount) = outputElementCount;

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
        Layer::ComputeHidden = [this](acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeHiddenPool(accel, fvBuffers, saturationCount); };

        Layer::ComputeConfig = [this](LayerConfiguration &layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t *saturationCount)
        {this->computeConfigPool(layerConfiguration, accel, fvBuffers, saturationCount); };
    }
}

void CnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    auto inputBuffer = layerConfiguration.InputBuffer
        ? layerConfiguration.InputBuffer->Get<int16_t>() : Input.Buffer;

    auto filterOutputBuffer = Activation ? Output.ScratchPad :
        (layerConfiguration.OutputBuffer ? layerConfiguration.OutputBuffer->Get<int32_t>() : Output.Buffer);

    if(!layerConfiguration.convolutionConfig)
        layerConfiguration.convolutionConfig = std::make_unique<ConvolutionConfig>(convolutionHiddenConfig);
    layerConfiguration.convolutionConfig->inputs = inputBuffer;
    layerConfiguration.convolutionConfig->convolutedOutputs = filterOutputBuffer;

    if (INTEL_NO_POOLING == Pooling.Type && Activation)
    {
        auto pwlOutputBuffer = layerConfiguration.OutputBuffer 
            ? layerConfiguration.OutputBuffer->Get<int16_t>() 
            : Output.Buffer;

        layerConfiguration.pwlOutputConfig = std::make_unique<PwlOutputConfig>(pwlOutputConfig);
        layerConfiguration.pwlOutputConfig->output = pwlOutputBuffer;
    }
}

void CnnLayer::computeHidden(acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    auto filterConfig = convolutionHiddenConfig;
    filterConfig.saturationCount = saturationCount;
    filterKernels.at(accel)(&filterConfig);
}

void CnnLayer::computeHiddenPwl(acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    computeHidden(accel, fvBuffers, saturationCount);

    auto pwlOutConfig = pwlOutputConfig;
    pwlOutConfig.saturationCount = saturationCount;
    pwlKernels.at(accel)(&pwlFilterConfig, fvBuffers->pwl, &pwlOutConfig);
}

void CnnLayer::computeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    auto filterConfig = *layerConfiguration.convolutionConfig;
    filterConfig.saturationCount = saturationCount;
    filterKernels.at(accel)(&filterConfig);
}

void CnnLayer::computeConfigPwl(const LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    computeConfig(layerConfiguration, accel, fvBuffers, saturationCount);

    auto pwlOutConfig = *layerConfiguration.pwlOutputConfig;
    pwlOutConfig.saturationCount = saturationCount;
    pwlKernels.at(accel)(&pwlFilterConfig, fvBuffers->pwl, &pwlOutConfig);
}

void CnnLayer::computeHiddenPool(acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    auto filterConfig = convolutionHiddenConfig;
    filterConfig.saturationCount = saturationCount;

    auto poolConfig = poolingHiddenConfig;
    poolConfig.buffer = fvBuffers->pool;

    poolingKernels.at(accel)(&filterConfig, &poolConfig, &pwlPoolConfig, fvBuffers->pwl);
}

void CnnLayer::computeConfigPool(LayerConfiguration& layerConfiguration, acceleration accel, KernelBuffers *fvBuffers, uint32_t * saturationCount) const
{
    auto filterConfig = *layerConfiguration.convolutionConfig;
    filterConfig.saturationCount = saturationCount;

    auto poolConfig = poolingHiddenConfig;
    poolConfig.buffer = fvBuffers->pool;

    poolingKernels.at(accel)(&filterConfig, &poolConfig, &pwlPoolConfig, fvBuffers->pwl);
}
