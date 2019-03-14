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

#include "Weight.h"

#include <memory>

#include "Expect.h"

using namespace GNA;

/* GNA_DATA_DISABLED may be supported in next generation */
static const DataModeLimits _ModesGen0_9 =
{
    { GNA_INT8, GNA_INT16, }, XNN_ERR_WEIGHT_BYTES
};

const FullCapabilitiesMap WeightTensor::capabilities =
{
    // TODO:3: add caps for previous device versions
    {INTEL_AFFINE, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits(
            {GNA_TENSOR_WH},    // W - #inputs, H - #outputs
            {{GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, XNN_ERR_WEIGHT_VOLUME}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_WEIGHT_VOLUME}}},
            _ModesGen0_9))},
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_W},    // W=H = #outputs
            {{GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, XNN_ERR_WEIGHT_VOLUME}}},
            _ModesGen0_9})}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_WH},   // W - #inputs, H - #outputs
            {{GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, XNN_ERR_WEIGHT_VOLUME}},
            {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, XNN_ERR_WEIGHT_VOLUME}}},
            _ModesGen0_9})}
    }},
    {INTEL_CONVOLUTIONAL, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NW},    // N - # filters, W - # filter coefficients
            {{GNA_DIM_N, {CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_N_FLT_COEFF_MPLY, CNN_ERR_CONV_FLT_COUNT}},
                {GNA_DIM_W, {CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, {2 * XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MPLY}, CNN_ERR_CONV_FLT_VOLUME}}},
            {{ GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR }, XNN_ERR_CONV_FLT_BYTES }})}
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_NHWD },    // N - # filters, HWD each filter dimensions
            {{GNA_DIM_N, {1, CNN_N_KERNELS_MAX, 1, CNN_ERR_CONV_FLT_COUNT}},
                {GNA_DIM_H, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, CNN_ERR_CONV_FLT_VOLUME}},
                {GNA_DIM_W, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MAX, 1, CNN_ERR_CONV_FLT_VOLUME}},
                {GNA_DIM_D, {CNN_N_KERNEL_ELEMENTS_PER_DIMENSION_MIN, XNN_N_IN_ELEMS_MAX, 1, CNN_ERR_CONV_FLT_VOLUME}}},
                // Padding to 16B is required for each Kernel
            {{ GNA_INT8, GNA_INT16, GNA_DATA_CONSTANT_SCALAR }, XNN_ERR_CONV_FLT_BYTES }})}
    }},
     {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_HDW },                  // H - GMM states, D - #mixtures, W - #inputs
            {{GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, GMM_BADNUMGMM}},
            {GNA_DIM_D, {1, GMM_MIXTURE_COMP_COUNT_MAX, 1, GMM_BADMIXCNUM}},
            {GNA_DIM_D, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, GNA_BADFEATLENGTH}}},
            { { GNA_INT8, GNA_INT16}, GMM_BADMODE},
            {GMM_MEM_ALIGNMENT, GMM_BADVARSALIGN}})}
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_WH },              // W - #inputs, H - #outputs
            {{GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, XNN_N_IN_ELEMS_MPLY, XNN_ERR_WEIGHT_VOLUME}},
                {GNA_DIM_H, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, XNN_ERR_WEIGHT_VOLUME}}},
            _ModesGen0_9})}
    }},
};


WeightTensor::WeightTensor(const Shape& dimensions, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn) :
    Tensor{ dimensions, dataMode, buffer, Validator{validatorIn, capabilities} }
{
};
