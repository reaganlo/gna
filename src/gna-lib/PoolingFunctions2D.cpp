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

#include "PoolingFunctions2D.h"

#include "AccelerationDetector.h"
#include "Expect.h"

using std::make_unique;
using std::unique_ptr;
using std::move;

using namespace GNA;

static const ComponentLimits __WH_limits=
{
    {GNA_TENSOR_WH},
    {{GNA_DIM_W, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorPoolStride}},
    {GNA_DIM_H, {1, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, Gna2StatusCnnErrorPoolStride}},}
};

const FullCapabilitiesMap PoolingFunction2D::windowLimits
{
    {INTEL_CONVOLUTIONAL_2D, {
        { GNA_3_0, std::make_shared<ComponentLimits>(__WH_limits)}
    }},
    //{GNA_LAYER_CNN_2D_POOLING, {
    //   { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
    //        {GNA_TENSOR_WHD},
    //        __WH_limits))}
    //}},
};

const FullCapabilitiesMap PoolingFunction2D::strideLimits
{
    {INTEL_CONVOLUTIONAL_2D, {
        { GNA_3_0, std::make_shared<ComponentLimits>(__WH_limits)}
    }},
    //{GNA_LAYER_CNN_2D_POOLING, {
    //    { GNA_3_0, std::make_shared<ComponentLimits>(ComponentLimits(
    //        {GNA_TENSOR_WHD},
    //        __WH_limits))}
    //}},
};

const SetLimits<nn_pool_type> PoolingFunction2D::typeLimits =
{
    { INTEL_MAX_POOLING, INTEL_SUM_POOLING }, Gna2StatusCnnErrorPoolType
};

const FullCapabilitiesMap PoolingFunction2D::outputCapabilities =
{
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},
            {{GNA_DIM_N, {1, CNN_N_KERNELS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes }})}
    }},
};

unique_ptr<PoolingFunction2D> PoolingFunction2D::Create(const TransformFactoryConfig& config)
{
    switch (config.validator.Operation)
    {
    case INTEL_CONVOLUTIONAL_2D:
    {
        auto cnn = static_cast<const nn_layer_cnn2d*>(config.layerDetails);
        auto pooling = nn_layer_pool2d{cnn->inputDimensions, cnn->pooling};
        return create(config, &pooling);
    }
    case GNA_LAYER_CNN_2D_POOLING:
    {
        auto pooling = static_cast<const nn_layer_pool2d*>(config.layerDetails);
        return create(config, pooling);
    }
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

std::unique_ptr<PoolingFunction2D> PoolingFunction2D::create(
    const TransformFactoryConfig& config, nn_layer_pool2d const * pool)
{
    if (INTEL_NO_POOLING != pool->pooling.type)
    {
        auto stride = make_unique<const Component>(pool->pooling.stride,
            Validator{ config.validator, strideLimits });
        auto window = make_unique<const Component>(pool->pooling.window,
            Validator{ config.validator, windowLimits });
        return make_unique<PoolingFunction2D>(
            BaseTransformConfig<PoolingKernel2D>{config,
                AccelerationDetector::GetKernelMap<PoolingKernel2D>(KERNEL_POOLING_2D, {config.input->Mode.Value})},
            pool->pooling.type, move(window), move(stride));
    }
    else
    {
        return unique_ptr<PoolingFunction2D>(nullptr);
    }
}

// unreachable code warning suppression
#if defined(_WIN32)
#pragma warning(disable : 702)
#endif
PoolingFunction2D::PoolingFunction2D(const BaseTransformConfig<PoolingKernel2D>& config,
    const nn_pool_type type, std::unique_ptr<const Component> window,
    std::unique_ptr<const Component> stride) :
    Transform{PoolingTransform2D, &config.kernels, config.input},
    Type{ type },
    Window{ move(window) },
    Stride{ move(stride) }
{
    Expect::InSet<nn_pool_type>(Type, typeLimits);
    Shape outputDims;
    outputDims[GNA_DIM_N] = Input->Dimensions.at(GNA_DIM_N);
    outputDims[GNA_DIM_D] = Input->Dimensions.at(GNA_DIM_D);

    for (const auto& iter : Stride->Dimensions)
    {
        auto const dim = iter.first;
        outputDims[dim] = 1
            + GnaCeilDiv(Input->Dimensions.at(dim) - Window->Dimensions.at(dim), iter.second);
    }

    Output = make_unique<Tensor>(outputDims, Input->Mode, config.outputBuffer,
        Validator{config.validator, outputCapabilities});

    auto out = Output->Dimensions;
    Expect::Fits(out, Input->Dimensions);

    auto configuration = nn_layer_pool2d{Input->Dimensions,
        {Type, Stride->Dimensions, Window->Dimensions}};

    hiddenConfig = make_unique<KernelConfig<PoolingConfig2D>>(PoolingConfig2D{configuration},
        BaseConfig{Input->Buffer, Output->Buffer});
}
