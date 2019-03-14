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

#include "LayerInput.h"

#include "Expect.h"

using namespace GNA;

static const std::vector<uint32_t> _Multipliers =
{ 2 * XNN_N_IN_ELEMS_MPLY, 1 * XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MPLY / 2};

static const ShapeLimits _NWLimits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_INPUT_VOLUME}},
    {GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, XNN_ERR_INPUT_VOLUME}}
};

static const DataModeLimits _ModesGen0_9 = {
    { GNA_INT16 },
    XNN_ERR_INPUT_BYTES
};

static const TensorLimits _NWTensorLimitsGen0_9 =
{
    {GNA_TENSOR_NW},
    _NWLimits,
    _ModesGen0_9
};

static const TensorLimits _WNTensorLimitsGen0_9 =
{
    {GNA_TENSOR_WN},
    _NWLimits,
    _ModesGen0_9
};

/* GNA_DATA_DISABLED may be supported in next generation */
static const DataModeLimits _ModesGen3 = {
    { GNA_INT8, GNA_INT16 },
    XNN_ERR_INPUT_BYTES
};

static const TensorLimits _NWTensorLimitsGen3 =
{
    {GNA_TENSOR_NW},
    _NWLimits,
    _ModesGen3
};

static const TensorLimits _WNTensorLimitsGen3 =
{
    {GNA_TENSOR_WN},
    _NWLimits,
    _ModesGen3
};

// GNA_TENSOR_WN -> GNA_DIM_W = columnCount
// GNA_TENSOR_WN -> GNA_DIM_N = rowCount
// GNA_TENSOR_NW -> GNA_DIM_W = rowCount
// GNA_TENSOR_NW -> GNA_DIM_N = columnCount
const FullCapabilitiesMap LayerInput::capabilities =
{
    {INTEL_AFFINE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_NWTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_NWTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_0_9, std::make_shared<TensorLimits>(_NWTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_NWTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(_NWTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_NWTensorLimitsGen3)}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_WN },
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, XNN_ERR_INPUT_VOLUME}}},
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_WN },
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, XNN_ERR_INPUT_VOLUME}}},
            _ModesGen3})}
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},    // N = 1
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_INPUT_VOLUME}}},
            _ModesGen3})}
    }},
   {GNA_LAYER_CNN_2D_POOLING, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},    // N = #kernels + GNA_BIAS_PER_KERNEL (HWD=1) or GNA_BIAS_PER_STRIDE (HWD each filter dimensions),
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_INPUT_VOLUME}}},
             { { GNA_INT8, GNA_INT16 }, XNN_ERR_INPUT_BYTES }})}
    }},
    {INTEL_COPY, {
        {GNA_0_9, std::make_shared<TensorLimits>(_WNTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_WNTensorLimitsGen3)}
    }},
    {INTEL_DEINTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_NWTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_NWTensorLimitsGen3)}
    }},
     {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_WN},                   // H - GMM states, D - #mixtures
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_INPUT_VOLUME}},
             {GNA_DIM_W, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, GNA_BADFEATLENGTH}}},
            { { GNA_INT8}, XNN_ERR_INPUT_BYTES }})}
    }},
     {INTEL_INTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_WNTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_WNTensorLimitsGen3)}
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(_WNTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_WNTensorLimitsGen3)}
    }}
};

LayerInput::LayerInput(const nn_layer &layer, const LayerValidator& validatorIn) :
    Tensor{GetDimensions(layer,  capabilities.GetOrder(validatorIn)),
        layer.nBytesPerInput, layer.pInputs,
        Validator{ validatorIn, capabilities } }
{
}

Shape LayerInput::GetDimensions(const nn_layer& layer, gna_tensor_order order)
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_CONVOLUTIONAL:
    case INTEL_COPY:
    case INTEL_INTERLEAVE:/* FALLTHRU */
    case INTEL_DEINTERLEAVE:
    case INTEL_GMM:
    case INTEL_RECURRENT:
        return {layer.nInputColumns, layer.nInputRows, order};
    case INTEL_CONVOLUTIONAL_2D:
    {
        auto const config = static_cast<nn_layer_cnn2d*>(layer.pLayerStruct);
        return {layer.nInputRows,
            config->inputDimensions.height,
            config->inputDimensions.width,
            config->inputDimensions.depth,
            order}; // GNA_TENSOR_NHWD
    }
    default:
        return {};
    }
}
