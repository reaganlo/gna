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

    auto biases = createBiasTensor(cnn->convolution, config.validator);

    return make_unique<ConvolutionFunction2D>(BaseTransformConfig<ConvolutionKernel2D>{config,
        AccelerationDetector::GetKernelMap<ConvolutionKernel2D>(
            KERNEL_CONVOLUTIONAL_2D,  { config.input->Mode, filters->Mode, (biases ? static_cast<gna_data_mode>(biases->Mode): GNA_DATA_DISABLED) })},
        move(filters), move(biases), move(stride), move(padding));
}

std::unique_ptr<const BiasTensor> ConvolutionFunction2D::createBiasTensor(
    gna_convolution_func const & convolution, const LayerValidator& validatorIn)
{
    Shape biasDims;
    switch (convolution.biases.mode)
    {
    case GNA_BIAS_PER_KERNEL:
    {
        biasDims = Shape(GNA_TENSOR_NHWD, convolution.filters.count, 1, 1, 1);
        break;
    }
    case GNA_BIAS_NOT_SUPPORTED:
    {
        biasDims = Shape();
    }
    case GNA_BIAS_PER_STRIDE:
    {
        biasDims = Shape(GNA_TENSOR_NHWD,
            1,
            convolution.filters.dimensions.height,
            convolution.filters.dimensions.width,
            convolution.filters.dimensions.depth);
        break;
    }
    default:
    {
        throw GnaException(XNN_ERR_BIAS_VOLUME);
    }
    }
    return make_unique<const BiasTensor>(
        biasDims,
        0,
        convolution.biases.dataMode,
        convolution.biases.biasesData,
        validatorIn,
        convolution.biases.mode);
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
    Shape outputDims;
    outputDims[GNA_DIM_N] = Input->Dimensions.at(GNA_DIM_N);
    // save #filters as Depth dimension of output (D in filters is used for 3D convolution)
    outputDims[GNA_DIM_D] = Filters->at(GNA_DIM_N);

    for (const auto& dimPair : Stride->Dimensions)
    {
        auto const dim = dimPair.first;
        outputDims[dim] =
            1 + (Input->Dimensions.at(dim) + (2 * Padding->Dimensions.at(dim)) - Filters->at(dim))
                 / dimPair.second;
    }

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

