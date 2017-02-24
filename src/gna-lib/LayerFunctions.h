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

#pragma once

#include <memory>

#include "common.h"
#include "Validator.h"

using std::unique_ptr;
using std::make_unique;

namespace GNA
{

typedef enum _WeightMode
{
    GNA_WEIGHT_1B = 1,
    GNA_WEIGHT_2B = 2,
} WeightMode;

struct Weight1B
{
    Weight1B(uint32_t size, const void *weights);
    ~Weight1B() = default;

    const uint8_t *Weights;
    static const WeightMode Mode = GNA_WEIGHT_1B;
};

struct Weight2B
{
public:
    Weight2B(uint32_t size, const void *weights);
    ~Weight2B() = default;

    const uint16_t *Weights;
    const WeightMode Mode = GNA_WEIGHT_2B;
};

struct BiasSimple
{
public:
    BiasSimple(uint32_t size, const void *biases);
    ~BiasSimple() = default;

    const nn_bias_s *Biases;
};

struct BiasCompound
{
public:
    BiasCompound(uint32_t size, const void *biases);
    ~BiasCompound() = default;

    const nn_bias_c *Biases;
};


// 2B Weights AffineFunction
class AffineFunctionSingle
{
public:
    ~AffineFunctionSingle() = default;

protected:
    AffineFunctionSingle(const nn_func_affine *affine);

    const nn_func_affine *sourceAffineFunction;
};

// 2B Weights AffineFunction
class AffineFunctionSingle2B : public AffineFunctionSingle, public Weight2B, public BiasSimple
{
public:
    AffineFunctionSingle2B(const nn_func_affine *affine);
    ~AffineFunctionSingle2B() = default;
};

// 1B Weights AffineFunction
class AffineFunctionSingle1B : public AffineFunctionSingle, public Weight1B, public BiasCompound
{
public:
    AffineFunctionSingle1B(const nn_func_affine *affine);
    ~AffineFunctionSingle1B() = default;
};


class AffineFunctionMutli : public BiasSimple
{
public:
    ~AffineFunctionMutli() = default;

    const uint32_t BiasVectorIndex;

protected:
    AffineFunctionMutli(const nn_func_affine_multi *affine);

    const nn_func_affine_multi *sourceAffineFunction; // TODO:KJ: remove when SW will use SoftwareModel classes only
};

// 2B Weights AffineFunction for Multi Bias
class AffineFunctionMulti2B : public AffineFunctionMutli, public Weight2B
{
public:
    AffineFunctionMulti2B(const nn_func_affine_multi *affine);
    ~AffineFunctionMulti2B() = default;

};

// 1B Weights AffineFunction for Multi Bias
class AffineFunctionMulti1B : public AffineFunctionMutli, public Weight1B
{
public:
    AffineFunctionMulti1B(const nn_func_affine_multi *affine);
    ~AffineFunctionMulti1B() = default;

    const nn_bias_c *WeightScaleFactors;
};


class ActivationFunction
{
public:
    static const uint32_t SegmentCountMax = XNN_N_PWL_SEGS_MAX;
    static const uint32_t SegmentCountMin = XNN_N_PWL_SEGS_MIN;

    ActivationFunction(const nn_func_pwl *pwl);
    virtual ~ActivationFunction() = default;

    const uint32_t SegmentCount;
    const nn_pwl_seg *Segments;
    const bool Enabled;

protected:
    const nn_func_pwl *sourcePwl;  // TODO:KJ: remove when SW will use SoftwareModel classes only
};

}
