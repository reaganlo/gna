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

#include "ConvolutionalFunctions2D.h"

#include "AccelerationDetector.h"
#include "Expect.h"

using std::make_unique;
using std::unique_ptr;
using std::move;

using namespace GNA;

const FullCapabilitiesMap ConvolutionFunction2D::strideLimits
{
    { INTEL_CONVOLUTIONAL_2D, {
        { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_WH},
            {{GNA_DIM_W, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, CNN_ERR_CONV_FLT_STRIDE}},
             {GNA_DIM_H, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, CNN_ERR_CONV_FLT_STRIDE}}}))}
    }}
};

const FullCapabilitiesMap ConvolutionFunction2D::paddingLimits
{
    { INTEL_CONVOLUTIONAL_2D, {
        { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_WH},
            {{GNA_DIM_W, {0, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, CNN_ERR_CONV_FLT_PADDING}},
             {GNA_DIM_H, {0, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, CNN_ERR_CONV_FLT_PADDING}}}))}
    }}
};

const FullCapabilitiesMap ConvolutionFunction2D::outputCapabilities
{
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, XNN_ERR_OUTPUT_BYTES }})}
    }},
};


unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::Create(const TransformFactoryConfig& config)
{
    switch (config.validator.Operation)
    {
    case INTEL_CONVOLUTIONAL_2D:
    {
        auto cnn = static_cast<const nn_layer_cnn2d*>(config.layerDetails);
        return create(config, cnn);
    }
    default:
        throw GnaException(XNN_ERR_LYR_OPERATION);
    }
}

std::unique_ptr<ConvolutionFunction2D> ConvolutionFunction2D::create(
    const TransformFactoryConfig& config, nn_layer_cnn2d const * cnn)
{
    auto filters = make_unique<const FiltersTensor>(
        Shape(GNA_TENSOR_NHWD, cnn->convolution.filters.count,
            cnn->convolution.filters.dimensions.height,
            cnn->convolution.filters.dimensions.width,
            cnn->convolution.filters.dimensions.depth),
        cnn->convolution.filters.dataMode,
        cnn->convolution.filters.filtersData,
        config.validator);

    auto stride = make_unique<const Component>(Shape(cnn->convolution.stride),
        Validator{ config.validator, strideLimits });

    auto padding = make_unique<const Component>(Shape(cnn->convolution.zeroPadding),
        Validator{ config.validator, paddingLimits });

    const Shape outputDims = GetOutputShape(config.input->Dimensions,
        filters->Dimensions, stride->Dimensions, padding->Dimensions);

    auto biases = CreateBiasTensor(cnn->convolution.biases, cnn->convolution.filters.count,
        outputDims, config.validator);

    return make_unique<ConvolutionFunction2D>(BaseTransformConfig<ConvolutionKernel2D>{config,
        AccelerationDetector::GetKernelMap<ConvolutionKernel2D>(
            KERNEL_CONVOLUTIONAL_2D,  { config.input->Mode, filters->Mode, (biases ? static_cast<gna_data_mode>(biases->Mode): GNA_DATA_DISABLED) })},
        move(filters), move(biases), move(stride), move(padding));
}

Shape ConvolutionFunction2D::CalculateBiasShape(const gna_bias_mode mode, const uint32_t filterCount, Shape const & outputShape)
{
    switch (mode)
    {
    case GNA_BIAS_PER_KERNEL:
    {
        return Shape(GNA_TENSOR_NHW, filterCount, 1, 1);
    }
    case GNA_BIAS_NOT_SUPPORTED:
    {
        return Shape();
    }
    case GNA_BIAS_PER_STRIDE:
    {
        return Shape(GNA_TENSOR_NHW,
            filterCount,
            outputShape.at(GNA_DIM_H),
            outputShape.at(GNA_DIM_W));
    }
    default:
    {
        return Shape{};
    }
    }
}

std::unique_ptr<const BiasTensor> ConvolutionFunction2D::CreateBiasTensor(
    gna_convolution_bias const & biases, uint32_t filtersCount,
    Shape const & outputShape, const LayerValidator& validatorIn)
{
    Shape biasDims = CalculateBiasShape(biases.mode, filtersCount, outputShape);
   
    return make_unique<const BiasTensor>(
        biasDims,
        0,
        biases.dataMode,
        biases.biasesData,
        validatorIn,
        biases.mode);
}

Shape ConvolutionFunction2D::GetOutputShape(Shape const & inputShape,
        Shape const & filerShape, Shape const & strideShape, Shape const & paddingShape)
{
    Shape outputShape;
    outputShape[GNA_DIM_N] = inputShape.at(GNA_DIM_N);
    // save #filters as Depth dimension of output (D in filters is used for 3D convolution)
    outputShape[GNA_DIM_D] = filerShape.at(GNA_DIM_N);

    for (const auto& dimPair : strideShape)
    {
        auto const dim = dimPair.first;
        outputShape[dim] =
            1 + (inputShape.at(dim) + (2 * paddingShape.at(dim)) - filerShape.at(dim))
            / dimPair.second;
    }
    return outputShape;
}

ConvolutionFunction2D::ConvolutionFunction2D(const BaseTransformConfig<ConvolutionKernel2D>& config,
    std::unique_ptr<const FiltersTensor> filters,
    std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<const Component> stride,
    std::unique_ptr<const Component> padding) :
    Transform{ConvolutionalTransform2D, &config.kernels, config.input},
    Biases{ move(biases) },
    Filters{ move(filters) },
    Stride{ move(stride) },
    Padding{ move(padding) }
{
    if (GNA_BIAS_PER_KERNEL == Biases->BiasMode)
    {
        Expect::Equal<uint32_t>(Biases->at(GNA_DIM_H), 1, XNN_ERR_BIAS_MODE);
        Expect::Equal<uint32_t>(Biases->at(GNA_DIM_W), 1, XNN_ERR_BIAS_MODE);
    }

    Shape outputDims = GetOutputShape(Input->Dimensions, Filters->Dimensions,
        Stride->Dimensions, Padding->Dimensions);

    Output = make_unique<Tensor>(outputDims, DataMode{GNA_INT32}, config.outputBuffer,
        Validator{config.validator, outputCapabilities});

    auto out = Output->Dimensions;
    out.erase(GNA_DIM_D);
    //Expect::Fits(out, Input->Dimensions); //TODO: Check if this check is valid/needed

    auto configuration = gna_convolution_func{
        {Filters->Mode, Filters->at(GNA_DIM_N), Filters->Dimensions, Filters->Buffer},
        Stride->Dimensions, Padding->Dimensions,
        {Biases->BiasMode, Biases->Mode, Biases->Buffer}};

    hiddenConfig = make_unique<KernelConfig<ConvolutionConfig2D>>(
        ConvolutionConfig2D{Input->Dimensions, configuration},
        BaseConfig{Input->Buffer, Output->Buffer});
}

