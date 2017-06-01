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

#include <map>
#include <memory>

#include "Address.h"
#include "common.h"
#include "pwl.h"
#include "XnnKernelApi.h"

namespace GNA
{

struct LayerConfiguration;

using Weight = Address<uint16_t * const>;
using Bias = Address<nn_bias_s * const>;

typedef enum _WeightMode
{
    GNA_WEIGHT_1B = 1,
    GNA_WEIGHT_2B = 0,
} WeightMode;

struct AffineBaseConfig
{
    AffineBaseConfig(uint32_t const outputElementCountIn, uint32_t const inputVectorCountIn,
        uint32_t const inputElementCountIn, int16_t const * inputIn, int32_t * const outputIn) :
        OutputElementCount{outputElementCountIn},
        InputVectorCount{inputVectorCountIn},
        InputElementCount{inputElementCountIn},
        Inputs{inputIn},
        Outputs{outputIn}
    {}
    uint32_t const OutputElementCount;
    uint32_t const InputVectorCount;
    uint32_t const InputElementCount;
    int16_t const * const Inputs;
    int32_t * const Outputs;
};

// AffineFunction interface
struct AffineFunction
{
public:
    static std::unique_ptr<const AffineFunction> Create(intel_layer_kind_t const kind, void const * layerDetails,
        AffineBaseConfig const & affineBase);

    std::unique_ptr<const AffineConfig> GetRunConfig(int16_t const * const inputs, int32_t * const outputs) const;

    void ComputeHidden(acceleration accel, uint32_t *saturationCount, KernelBuffers *fvBuffers) const;
    virtual void ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, uint32_t *saturationCount,
        KernelBuffers *fvBuffers) const;

    const WeightMode Mode;
    const Weight Weights;
    const Bias Biases;

protected:
    AffineFunction(const std::map<const acceleration, const AffineKernel>& kernels, const WeightMode Mode,
        const Weight weights, const Bias biases);

    const std::map<const acceleration, const AffineKernel>&  kernels;
    std::unique_ptr<const AffineConfig> hiddenConfig;
};

// 2B Weights AffineFunction
class AffineFunctionSingle : public AffineFunction
{
public:
    ~AffineFunctionSingle() = default;

    virtual void ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, uint32_t *saturationCount,
        KernelBuffers *fvBuffers) const override;

protected:
    AffineFunctionSingle(const std::map<const acceleration, const AffineKernel>& kernels,
        const std::map<const acceleration, const AffineActiveListKernel>& kernelsAl, const WeightMode Mode, 
        const Weight weights, const Bias biases);

    const std::map<const acceleration, const AffineActiveListKernel>& kernelsAl;
};

// 2B Weights AffineFunction
class AffineFunctionSingle2B : public AffineFunctionSingle
{
public:
    AffineFunctionSingle2B(const nn_func_affine *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernels,
        const std::map<const acceleration, const AffineActiveListKernel>& kernelsAl);
    ~AffineFunctionSingle2B() = default;
};

// 1B Weights AffineFunction
class AffineFunctionSingle1B : public AffineFunctionSingle
{
public:
    AffineFunctionSingle1B(const nn_func_affine *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernels,
        const std::map<const acceleration, const AffineActiveListKernel>& kernelsAl);
    ~AffineFunctionSingle1B() = default;
};


class AffineFunctionMulti : public AffineFunction
{
public:
    ~AffineFunctionMulti() = default;

    const uint32_t BiasVectorCount;
    const uint32_t BiasVectorIndex;

    const nn_bias_s * const GetMultibias() const;

protected:
    AffineFunctionMulti(const nn_func_affine_multi *affine,
        const std::map<const acceleration, const AffineKernel>& kernels, const WeightMode Mode,
        const Weight weights, const Bias biases);
};

// 2B Weights AffineFunction for Multi Bias
class AffineFunctionMulti2B : public AffineFunctionMulti
{
public:
    AffineFunctionMulti2B(const nn_func_affine_multi *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernels);
    ~AffineFunctionMulti2B() = default;
};

// 1B Weights AffineFunction for Multi Bias
class AffineFunctionMulti1B : public AffineFunctionMulti
{
public:
    AffineFunctionMulti1B(const nn_func_affine_multi *affine, AffineBaseConfig const & affineBase,
        const std::map<const acceleration, const AffineKernel>& kernels);
    ~AffineFunctionMulti1B() = default;

    const nn_bias_c *WeightScaleFactors;
};

class ActivationFunction
{
public:
    static const std::unique_ptr<const ActivationFunction> Create(const nn_func_pwl * const pwl, const bool mandatory,
        int32_t const * const Inputs, const PwlOutputConfig& outputConfig);

    static const uint32_t SegmentCountMax = XNN_N_PWL_SEGS_MAX;
    static const uint32_t SegmentCountMin = XNN_N_PWL_SEGS_MIN;

    ActivationFunction(const nn_func_pwl * const pwl, int32_t const * const inputIn, const PwlOutputConfig& outputConfig);
    ActivationFunction() = delete;
    virtual ~ActivationFunction() = default;

    std::unique_ptr<PwlOutputConfig> GetOutputConfig(int16_t * const outputs) const;
    void ComputeHidden(acceleration accel, uint32_t *saturationCount) const;
    void ComputeConfig(const LayerConfiguration& layerConfiguration, acceleration accel, uint32_t *saturationCount) const;
    

    uint32_t const SegmentCount;
    nn_pwl_seg const * const Segments;
    PwlCached const Pwl;

private:
    const std::map<const acceleration, const PwlKernel>& Kernels;

    PwlOutputConfig const OutputConfig;
};

}
