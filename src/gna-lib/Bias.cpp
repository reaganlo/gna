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

#include "Capabilities.h"
#include "Expect.h"
#include "ParameterLimits.h"
#include "PoolingFunctions2D.h"
#include "Shape.h"
#include "Validator.h"

#include "ConvolutionKernelArguments.h"

#include "gna2-common-api.h"
#include "common.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <cstdint>
#include <map>
#include <memory>
#include <utility>

using namespace GNA;

static const DataModeLimits _ModesGen0_9 = {
    { GNA_INT32 },
    Gna2StatusXnnErrorBiasBytes };

static const DataModeLimits _ModesWithRichGen0_9 = {
    { GNA_INT32, GNA_DATA_RICH_FORMAT },
    Gna2StatusXnnErrorBiasBytes };

static const DataModeLimits _ModesGen3 = {
    { GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_DISABLED },
    Gna2StatusXnnErrorBiasBytes };

static const DataModeLimits _ModesWithRichGen3 = {
    { GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_DISABLED, GNA_DATA_RICH_FORMAT },
    Gna2StatusXnnErrorBiasBytes };

const FullCapabilitiesMap BiasTensor::capabilities =
{
    {INTEL_AFFINE, {
        {GNA_0_9,std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesWithRichGen0_9}),
        },
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesWithRichGen3})}
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesWithRichGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesWithRichGen3})}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {
                {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
                {GNA_DIM_W, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorBiasVolume}}
            },
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
            {GNA_DIM_W, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesGen3})}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_N},          // H - #kernel (GNA_BIAS_PER_KERNEL)
            {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusXnnErrorBiasVolume}}},
            _ModesGen0_9})},
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHW},          // N - #kernel (GNA_BIAS_PER_KERNEL)
            {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, Gna2StatusXnnErrorBiasVolume}},
                {GNA_DIM_H, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}},
                {GNA_DIM_W, {1, 1, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHW},    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1) or GNA_BIAS_PER_STRIDE (HW conv. out dimensions),
                {{GNA_DIM_N, {1, CNN_N_FLT_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
                {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}},
                {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            _ModesGen3})}
    }},
    {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HD},                   // H - GMM states, D - #mixtures
            {{GNA_DIM_H, {1, GMM_MIXTURE_COMP_COUNT_MAX, 1, Gna2StatusGmmBadMixCnum}},
                {GNA_DIM_D, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}}},
            { {GNA_INT32}, Gna2StatusGmmBadMode},
            {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadGconstAlign}})}
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, Gna2StatusXnnErrorBiasVolume}}},
            _ModesWithRichGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, Gna2StatusXnnErrorBiasVolume}}},
            _ModesWithRichGen3})}
    }}
};

const SetLimits<KernelBiasMode> BiasTensor::modeLimits
{
    { KernelBiasModeDisabled, KernelBiasModePerFilter, KernelBiasModePerStride },
    Gna2StatusXnnErrorBiasMode
};

BiasTensor::BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn, Gna2BiasMode biasMode) :
    Tensor{ dimensions, dataMode, buffer, Validator{ validatorIn, capabilities } },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, dataMode.Mode) }
{
    validate();
}

BiasTensor::BiasTensor(const Gna2Tensor &apiTensor, const uint32_t biasVectorIndex,
        Gna2BiasMode biasMode, const LayerValidator& validatorIn) :
    Tensor{ apiTensor, Validator { validatorIn, capabilities } },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, apiTensor.Mode) }
{
    validate();
}

void BiasTensor::validate() const
{
    const auto vectorCountIter = Dimensions.find(GNA_DIM_W);
    auto vectorCount = ui32_1;
    if (Dimensions.end() != vectorCountIter)
    {
        vectorCount = vectorCountIter->second;
    }
    Expect::InRange(VectorIndex, ui32_0, vectorCount - 1, Gna2StatusXnnErrorBiasIndex);

    Expect::InSet(BiasMode, modeLimits);
}

KernelBiasMode BiasTensor::ToKernelBiasMode(Gna2BiasMode mode, Gna2TensorMode tensorMode)
{
    //TODO:3:Handle constant scalar when enabled in HW
    if (Gna2TensorModeDisabled == tensorMode ||
        Gna2TensorModeConstantScalar == tensorMode)
    {
        return KernelBiasModeDisabled;
    }
    static const std::map<Gna2BiasMode, KernelBiasMode> biasMap
    {
        { Gna2BiasModeDefault, KernelBiasModePerFilter },
        { Gna2BiasModePerStride, KernelBiasModePerStride },
        { Gna2BiasModeGrouping, KernelBiasModePerFilter },
    };
    return biasMap.at(mode);
}
