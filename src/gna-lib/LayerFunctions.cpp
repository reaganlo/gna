/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include "LayerFunctions.h"
#include "Validator.h"

using namespace GNA;

Weight1B::Weight1B(uint32_t size, const void *weights) :
    Weights(static_cast<const uint8_t*>(weights))
{
    Expect::ValidBuffer(Weights);
    Expect::True(sizeof(uint8_t) == size, XNN_ERR_WEIGHT_BYTES);
}

Weight2B::Weight2B(uint32_t size, const void *weights) :
    Weights(static_cast<const uint16_t*>(weights))
{
    Expect::ValidBuffer(Weights);
    Expect::True(sizeof(uint16_t) == size, XNN_ERR_WEIGHT_BYTES);
}

BiasSimple::BiasSimple(uint32_t size, const void *biases) :
    Biases(static_cast<const nn_bias_s*>(biases))
{
    Expect::ValidBuffer(Biases);
    Expect::True(sizeof(nn_bias_s) == size, XNN_ERR_BIAS_BYTES);
}

BiasCompound::BiasCompound(uint32_t size, const void *biases) :
    Biases(static_cast<const nn_bias_c*>(biases))
{
    Expect::ValidBuffer(Biases);
    Expect::True(sizeof(nn_bias_c) == size, XNN_ERR_BIAS_BYTES);
}


AffineFunctionSingle::AffineFunctionSingle(const nn_func_affine * affine) :
    sourceAffineFunction(static_cast<const nn_func_affine *>(affine))
{
    Expect::NotNull(sourceAffineFunction);
}

AffineFunctionSingle2B::AffineFunctionSingle2B(const nn_func_affine *affine) :
    AffineFunctionSingle(affine),
    Weight2B(affine->nBytesPerWeight, affine->pWeights),
    BiasSimple(affine->nBytesPerBias, affine->pBiases)
{
}

AffineFunctionSingle1B::AffineFunctionSingle1B(const nn_func_affine *affine) :
    AffineFunctionSingle(affine),
    Weight1B(affine->nBytesPerWeight, affine->pWeights),
    BiasCompound(affine->nBytesPerBias, affine->pBiases)
{
}


AffineFunctionMutli::AffineFunctionMutli(const nn_func_affine_multi *affine) :
    BiasSimple(sizeof(nn_bias_s), affine->pBiases),
    BiasVectorIndex(affine->biasVectorIndex),
    sourceAffineFunction(affine)
{
    Expect::InRange(BiasVectorIndex, 0, XNN_N_GROUP_MAX - 1, XNN_ERR_GROUPING);
    Expect::NotNull(sourceAffineFunction);
}

AffineFunctionMulti2B::AffineFunctionMulti2B(const nn_func_affine_multi *affine) :
    AffineFunctionMutli(affine),
    Weight2B(affine->nBytesPerWeight, affine->pWeights)   
{
}

AffineFunctionMulti1B::AffineFunctionMulti1B(const nn_func_affine_multi *affine) :
    AffineFunctionMutli(affine),
    Weight1B(affine->nBytesPerWeight, affine->pWeights),
    WeightScaleFactors(affine->weightScaleFactors)
{
    Expect::ValidBuffer(WeightScaleFactors);
}

ActivationFunction::ActivationFunction(const nn_func_pwl *pwl) :
    SegmentCount(static_cast<bool>(pwl->nSegments)),
    Segments(static_cast<nn_pwl_seg*>(pwl->pSegments)),
    Enabled((nullptr != pwl->pSegments) && (pwl->nSegments > 0)),
    sourcePwl(static_cast<const nn_func_pwl*>(pwl))
{
    if(Enabled)
    {
        Expect::ValidBuffer(Segments, XNN_ERR_PWL_DATA);
        Expect::InRange(SegmentCount, SegmentCountMin, SegmentCountMax, XNN_ERR_PWL_SEGMENTS);
    }
}

