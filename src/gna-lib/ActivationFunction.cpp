/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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
#include "Address.h"
#include "Capabilities.h"
#include "GnaException.h"
#include "ParameterLimits.h"
#include "Shape.h"
#include "Validator.h"

#include "gna2-common-api.h"

#include "gna-api.h"

#include <algorithm>
#include <memory>

using namespace GNA;

static const TensorLimits _ActivationLimitsGen0_9 =
{
    {{GNA_TENSOR_W},    // W - #inputs, H - #outputs
    {{GNA_DIM_W, {XNN_N_PWL_SEGS_MIN, XNN_N_PWL_SEGS_MAX, 1, Gna2StatusXnnErrorPwlSegments}}}},
    {{ GNA_DATA_RICH_FORMAT },
    Gna2StatusXnnErrorOutputBytes}
};

const FullCapabilitiesMap ActivationFunction::capabilities =
{
    {INTEL_AFFINE,{
        {GNA_0_9, std::make_shared<TensorLimits>(_ActivationLimitsGen0_9)},
    }},
    {INTEL_AFFINE_DIAGONAL,{
        {GNA_0_9, std::make_shared<TensorLimits>(_ActivationLimitsGen0_9)},
    }},
    {INTEL_AFFINE_MULTIBIAS,{
        {GNA_2_0, std::make_shared<TensorLimits>(_ActivationLimitsGen0_9)},
    }},
    {INTEL_CONVOLUTIONAL,{
        {GNA_1_0, std::make_shared<TensorLimits>(_ActivationLimitsGen0_9)},
    }},
    {INTEL_CONVOLUTIONAL_2D,{
        {GNA_3_0, std::make_shared<TensorLimits>(_ActivationLimitsGen0_9)}
    }},
    {INTEL_RECURRENT,{
        {GNA_0_9, std::make_shared<TensorLimits>(_ActivationLimitsGen0_9)},
    }}
};

// TODO:3: Copy of LayerOuputCapabilities due to unsolved discrepancy in layer architecture
static const ShapeLimits _FlatLimits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
    {GNA_DIM_H, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, Gna2StatusXnnErrorOutputVolume}}
};

static const ShapeLimits _InterleaveLimits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
    {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}
};

static const DataModeLimits _ModesGen0_9 =
{
    {GNA_INT16, GNA_DATA_ACTIVATION_DISABLED},
    Gna2StatusXnnErrorOutputBytes
};

static const TensorLimits _InterleaveTensorLimitsGen0_9 =
{
    {GNA_TENSOR_NH},
    _InterleaveLimits,
    _ModesGen0_9
};

static const TensorLimits _FlatTensorLimitsGen0_9 =
{
    {GNA_TENSOR_HN},
    _FlatLimits,
    _ModesGen0_9
};

static const DataModeLimits _ModesGen3 =
{
    {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
    Gna2StatusXnnErrorOutputBytes
};

static const TensorLimits _InterleaveTensorLimitsGen3 =
{
    {GNA_TENSOR_NH},
    _InterleaveLimits,
    _ModesGen3
};

static const TensorLimits _FlatTensorLimitsGen3 =
{
    {GNA_TENSOR_HN},
    _FlatLimits,
    _ModesGen3
};

const FullCapabilitiesMap ActivationFunction::outputCapabilities =
{
    {INTEL_AFFINE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_HN },
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_HN },
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            _ModesGen3})},
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            {{GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED}, Gna2StatusXnnErrorOutputBytes }})}
    }},
    {INTEL_COPY, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
    {INTEL_DEINTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
    {INTEL_INTERLEAVE, {
        { GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9) },
        { GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3) }
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN},
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, Gna2StatusXnnErrorOutputVolume}}}, // must be multiple 32 to keep 64B output buffer alignment
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN},
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, Gna2StatusXnnErrorOutputVolume}}}, // must be multiple 32 to keep 64B output buffer alignment
            _ModesGen3})}
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
        {
            mandatory = true;
        }
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
        auto pwlFunction = std::make_unique<Tensor>(Shape(GNA_TENSOR_W, pwl->nSegments),
            GNA_DATA_RICH_FORMAT, pwl->pSegments, Validator{config.validator, capabilities});
        return std::make_unique<ActivationFunction>(
            BaseTransformConfig<ActivationKernel>{config,
            AccelerationDetector::GetKernelMap<ActivationKernel>(KERNEL_PWL)}, config.outputMode,
            std::move(pwlFunction));
    }

    auto valuePtr = &(config.output->Mode.Value);
    *((gna_data_mode*)valuePtr) = GNA_DATA_ACTIVATION_DISABLED;
    return std::unique_ptr<ActivationFunction>(nullptr);
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
        throw GnaException{ Gna2StatusXnnErrorLyrOperation };
    }
}

PwlCached ActivationFunction::createPwlCached(const gna_data_mode mode,
    nn_pwl_seg const * const segmentsIn, uint32_t segmentCountIn)
{
    try
    {
        return PwlCached(mode, segmentsIn, segmentCountIn);
    }
    catch (const std::runtime_error&)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
}

ActivationFunction::ActivationFunction(const BaseTransformConfig<ActivationKernel>& config,
    DataMode mode, std::unique_ptr<Tensor> pwl) :
    Transform{ActivationTransform, &config.kernels, config.input},
    Segments{ std::move(pwl) },
    Pwl{ createPwlCached(config.outputMode, Segments->Buffer, Segments->Count) }
{
    const auto validator = Validator{config.validator, outputCapabilities};
    Output = std::make_unique<Tensor>(config.input->Dimensions, mode, config.outputBuffer,
        validator);

    hiddenConfig = std::make_unique<KernelConfig<ActivationConfig>>(
        ActivationConfig{Output->Count, &Pwl}, BufferMap{Input->Buffer, config.outputBuffer});
}
