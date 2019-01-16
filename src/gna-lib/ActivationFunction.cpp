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

#include "ActivationFunction.h"

#include "AccelerationDetector.h"
#include "DeviceLayerSupport.h"
#include "LayerConfiguration.h"

using namespace GNA;

static const ComponentLimits _ActivationLimitsBase = {
    {GNA_TENSOR_W},    // W - #inputs, H - #outputs
    {{GNA_DIM_W, {XNN_N_PWL_SEGS_MIN, XNN_N_PWL_SEGS_MAX, 1, XNN_ERR_PWL_SEGMENTS}}}
};
static const TensorLimits _ActivationLimits = {
   _ActivationLimitsBase,
    {{ GNA_INT16 }, XNN_ERR_OUTPUT_BYTES}
};
static const TensorLimits _ActivationLimits_3 = {
   _ActivationLimitsBase,
    {{ GNA_INT8, GNA_INT16, GNA_INT32 }, XNN_ERR_OUTPUT_BYTES}
};

const FullCapabilitiesMap ActivationFunction::capabilities =
{
    {INTEL_AFFINE,{
        {GNA_0_9, std::make_shared<TensorLimits>(_ActivationLimits)}, // TODO:3: verify if shared_ptr here releases memory correctly on exit
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimits_3)}
    }},
    {INTEL_AFFINE_DIAGONAL,{
        {GNA_0_9, std::make_shared<TensorLimits>(_ActivationLimits)},
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimits_3)}
    }},
    {INTEL_AFFINE_MULTIBIAS,{
        {GNA_2_0, std::make_shared<TensorLimits>(_ActivationLimits)},
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimits_3)}
    }},
    {INTEL_CONVOLUTIONAL,{
        {GNA_1_0, std::make_shared<TensorLimits>(_ActivationLimits)},
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimits_3)}
    }},
    {INTEL_CONVOLUTIONAL_2D,{
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimits_3)}
    }},
    {INTEL_RECURRENT,{
        {GNA_0_9, std::make_shared<TensorLimits>(_ActivationLimits)},
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimits_3)}
    }}
};


// TODO:3: Copy of LayerOuputCapabilities due to unsolved discrepancy in layer architecture
static const ShapeLimits __flat_limits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
    {GNA_DIM_H, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, XNN_ERR_OUTPUT_VOLUME}}
};

static const ShapeLimits __interleave_limits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
    {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}
};

static const DataModeLimits __modes = { {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
    XNN_ERR_OUTPUT_BYTES };

static const TensorLimits __interleave_tensor_limits =
{
    {GNA_TENSOR_NH},
    __interleave_limits,
    __modes
};

static const TensorLimits __flat_tensor_limits =
{
    {GNA_TENSOR_HN},
    __flat_limits,
    __modes
};

const FullCapabilitiesMap ActivationFunction::outputCapabilities =
{
    // TODO:3: add caps for previous device versions
    {INTEL_AFFINE, {
        {GNA_3_0, std::make_shared<TensorLimits>(__interleave_tensor_limits)}
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_3_0, std::make_shared<TensorLimits>(__interleave_tensor_limits)}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_3_0, std::make_shared<TensorLimits>(__interleave_tensor_limits)}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_HN },
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            __modes})}
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, XNN_ERR_OUTPUT_BYTES }})}
    }},
    {INTEL_COPY, {
        {GNA_3_0, std::make_shared<TensorLimits>(__flat_tensor_limits)}
    }},
    {INTEL_DEINTERLEAVE, {
        {GNA_3_0, std::make_shared<TensorLimits>(__flat_tensor_limits)}
    }},
     {INTEL_GMM, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN}, // H - GMM States, N - grouping
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            { { GNA_INT32, GNA_DATA_ACTIVATION_DISABLED }, XNN_ERR_OUTPUT_BYTES }})}
    }},
    {INTEL_INTERLEAVE, {
        { GNA_3_0, std::make_shared<TensorLimits>(__interleave_tensor_limits) }
    }},
    {INTEL_RECURRENT, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN},
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, XNN_ERR_OUTPUT_VOLUME}}}, // must be multiple 32 to keep 64B output buffer alignment
            __modes})}
    }}
};
// end of copy

std::unique_ptr<ActivationFunction> ActivationFunction::Create(const TransformFactoryConfig& config)
{
    auto mandatory = false; // TODO:3: use CAPS to determine
    switch (config.validator.Operation)
    {
    case INTEL_CONVOLUTIONAL:
    {
        auto cnn = static_cast<nn_layer_conv const*>(config.layerDetails);
        if (INTEL_NO_POOLING != cnn->poolType)
            mandatory = true;
        break;
    }
    case GNA_LAYER_CNN_2D_POOLING:
    {
        return std::unique_ptr<ActivationFunction>(nullptr);
    }
    case INTEL_RECURRENT:
        mandatory = true;
        break;
    default:
        break;
    }

    const nn_func_pwl *pwl = getPwl(config.layerDetails, config.validator.Operation);

    if (mandatory || IsEnabled(pwl))
    {
        auto pwlFunction = std::make_unique<Tensor>(Shape(0, pwl->nSegments, 0),
            config.outputMode, pwl->pSegments, Validator{config.validator, capabilities});
        return std::make_unique<ActivationFunction>(
            BaseTransformConfig<ActivationKernel>{config,
            AccelerationDetector::GetKernelMap<ActivationKernel>(KERNEL_PWL)}, config.outputMode,
            std::move(pwlFunction));
    }
    else
    {
        // TODO:3: Rethink what to do with backward compatibility (nOuputBytes=GNA_INT32)
        Expect::InSet(config.outputMode, { GNA_INT32, GNA_DATA_ACTIVATION_DISABLED }, XNN_ERR_OUTPUT_BYTES);
        return std::unique_ptr<ActivationFunction>(nullptr);
    }
}

nn_func_pwl const * ActivationFunction::getPwl(void const *layerDetails, nn_operation operation)
{
    switch (operation)
    {
    case INTEL_AFFINE: /* FALLTHRU */
    case INTEL_AFFINE_DIAGONAL:
        return &static_cast<nn_layer_affine const*>(layerDetails)->pwl;
    case INTEL_AFFINE_MULTIBIAS:
        return &static_cast<nn_layer_affine_multi const*>(layerDetails)->pwl;
    case INTEL_CONVOLUTIONAL:
        return &static_cast<nn_layer_conv const*>(layerDetails)->pwl;
    case INTEL_CONVOLUTIONAL_2D:
        return &static_cast<nn_layer_cnn2d const*>(layerDetails)->activation;
    case INTEL_RECURRENT:
        return &static_cast<nn_layer_reccurent const*>(layerDetails)->pwl;
    default:
        throw GnaException{ XNN_ERR_LYR_OPERATION };
    }
}

ActivationFunction::ActivationFunction(const BaseTransformConfig<ActivationKernel>& config,
    DataMode mode, std::unique_ptr<Tensor> pwl) :
    Transform{ActivationTransform, &config.kernels, config.input},
    Segments{ std::move(pwl) },
    Pwl{ Segments->Mode, Segments->Buffer, Segments->Count }
{
    const auto validator = Validator{config.validator, outputCapabilities};
    Output = std::make_unique<Tensor>(config.input->Dimensions, mode, config.outputBuffer,
        validator);

    hiddenConfig = std::make_unique<KernelConfig<ActivationConfig>>(
        ActivationConfig{Output->Count, &Pwl}, BufferMap{Input->Buffer, config.outputBuffer});
}
