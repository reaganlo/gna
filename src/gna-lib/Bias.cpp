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

#include "Bias.h"

#include "Expect.h"

using namespace GNA;

static const DataModeLimits Modes = {
    { GNA_INT8, GNA_INT16, GNA_INT32/*, GNA_DATA_DISABLED */},
    XNN_ERR_BIAS_BYTES };

static const DataModeLimits ModesWithRich = {
    { GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_RICH_FORMAT },
    XNN_ERR_BIAS_BYTES };

const FullCapabilitiesMap BiasTensor::capabilities =
{
    // TODO:3: add caps for previous device versions
    {INTEL_AFFINE, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}}},
            ModesWithRich})}
        //{GNA_0_9,std::make_shared<TensorLimits>() // TODO:3:cover backward compatibility in all components
        //},
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}}},
            ModesWithRich})}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NH},
                {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, XNN_ERR_BIAS_VOLUME}},
                {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}}},
            Modes})}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},          // H - #kernel (GNA_BIAS_PER_KERNEL)
            {{GNA_DIM_H, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, XNN_ERR_BIAS_VOLUME}}},
            Modes})}
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},    // N = #kernels + GNA_BIAS_PER_KERNEL (HWD=1) or GNA_BIAS_PER_STRIDE (HWD each filter dimensions),
                {{GNA_DIM_N, {1, CNN_N_FLT_MAX, 1, XNN_ERR_BIAS_VOLUME}},
                {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}},
                {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}},
                {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_BIAS_VOLUME}},},
            Modes})}
    }},
     {INTEL_GMM, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HD},                   // H - GMM states, D - #mixtures
            {{GNA_DIM_H, {1, GMM_MIXTURE_COMP_COUNT_MAX, 1, GMM_BADMIXCNUM}},
                {GNA_DIM_D, {1, GMM_STATES_COUNT_MAX, 1, GMM_BADNUMGMM}}},
            { {GNA_INT32}, GMM_BADMODE},
            {GMM_MEM_ALIGNMENT, GMM_BADGCONSTALIGN}})}
    }},
    {INTEL_RECURRENT, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, XNN_ERR_BIAS_VOLUME}}},
            ModesWithRich})}
    }}
};

const SetLimits<gna_bias_mode> BiasTensor::modeLimits
{
    { GNA_BIAS_PER_KERNEL, GNA_BIAS_PER_STRIDE }, XNN_ERR_BIAS_MODE
};

BiasTensor::BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn, gna_bias_mode mode) :
    Tensor{ dimensions, dataMode, buffer, Validator{ validatorIn, capabilities } },
    VectorIndex{ biasVectorIndex },
    BiasMode { mode }
{
    auto vectorCountIter = Dimensions.find(GNA_DIM_N);
    auto vectorCount = 1;
    if (Dimensions.end() != vectorCountIter)
    {
        vectorCount = vectorCountIter->second;
    }
    Expect::InRange<uint32_t>(VectorIndex, 0, vectorCount - 1, XNN_ERR_BIAS_INDEX);
    Expect::InSet(BiasMode, modeLimits);
};

