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

#include "LayerOutput.h"

#include "Expect.h"

using namespace GNA;

static const ShapeLimits _FlatLimits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
    {GNA_DIM_H, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, XNN_ERR_OUTPUT_VOLUME}}
};

static const ShapeLimits _InterleaveLimits =
{
    {GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
    {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}
};

static const DataModeLimits _ModesGen0_9 =
{
    {GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
    XNN_ERR_OUTPUT_BYTES
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
    XNN_ERR_OUTPUT_BYTES
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

const FullCapabilitiesMap LayerOutput::capabilities =
{
    // TODO:3: add caps for previous device versions
    {INTEL_AFFINE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_HN },
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_HN },
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            _ModesGen3})}
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_WN},
            {{GNA_DIM_N, {1, 1, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_W,
                {1, XNN_N_IN_ELEMS_MAX*XNN_N_IN_ELEMS_MAX/**XNN_N_IN_ELEMS_MAX*/, 1, XNN_ERR_OUTPUT_VOLUME}}},
                // use as NxHxW
            _ModesGen3})}
    }},
    {INTEL_COPY, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
    {INTEL_DEINTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
     {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN}, // H - GMM States, N - grouping
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, XNN_ERR_OUTPUT_VOLUME}}},
            { { GNA_INT32, GNA_DATA_ACTIVATION_DISABLED }, XNN_ERR_OUTPUT_BYTES }})}
    }},
    {INTEL_INTERLEAVE, {
        { GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9) },
        { GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3) }
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN},
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, XNN_ERR_OUTPUT_VOLUME}}}, // must be multiple 32 to keep 64B output buffer alignment
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN},
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_OUTPUT_VOLUME}},
             {GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, XNN_ERR_OUTPUT_VOLUME}}}, // must be multiple 32 to keep 64B output buffer alignment
            _ModesGen3})}
    }}
};

LayerOutput::LayerOutput(const nn_layer& layer, const LayerValidator& validatorIn) :
    Tensor{ { layer.nOutputColumns, layer.nOutputRows, capabilities.GetOrder(validatorIn) },
        layer.nBytesPerOutput, layer.pOutputs,
        Validator{ validatorIn, capabilities } },
    ScratchPad{ Dimensions,
        layer.nBytesPerIntermediateOutput, layer.pOutputsIntermediate,
        Validator{ validatorIn, capabilities } }
{
    Expect::True(GNA_INT32 == ScratchPad.Mode, XNN_ERR_INT_OUTPUT_BYTES);
    //Expect::ValidBuffer(ScratchPad); // TODO: review when scratch-pad is allocated by gna-lib
};
