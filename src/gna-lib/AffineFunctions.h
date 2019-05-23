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

#pragma once

#include "Address.h"
#include "Bias.h"
#include "KernelArguments.h"
#include "Tensor.h"
#include "Weight.h"
#include "XnnKernel.h"

#include "gna2-inference-impl.h"

#include <cstdint>
#include <map>
#include <memory>

namespace GNA
{

class FullCapabilitiesMap;
class LayerValidator;

struct LayerConfiguration;

// AffineFunction interface
struct AffineFunction
{
public:
    virtual ~AffineFunction() = default;

    // dimensions: NWH
    static std::unique_ptr<const AffineFunction> Create(const Tensor* input, const Tensor* output,
        void const * layerDetails, const LayerValidator& validatorIn);

    std::unique_ptr<const AffineConfig> GetRequestConfig(const BaseAddress& inputs,
        const BaseAddress& outputs) const;

    void ComputeHidden(AccelerationMode accel, ExecutionConfig const & execution) const;
    virtual void Compute(const LayerConfiguration& layerConfiguration,
        AccelerationMode accel, ExecutionConfig const & execution) const;

    std::unique_ptr<const WeightTensor> Weights;
    std::unique_ptr<const BiasTensor> Biases;

protected:
    AffineFunction(const KernelMap<AffineKernel>& kernelsIn,
        std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases);

    const KernelMap<AffineKernel>&  kernels;
    std::unique_ptr<const AffineConfig> hiddenConfig;
};

class AffineFunctionSingle : public AffineFunction
{
public:
    AffineFunctionSingle(const BaseAddress& input, const BaseAddress& output, const uint32_t vectorCount,
        std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
        const KernelMap<AffineKernel>& kernelsIn,
        const KernelMap<AffineActiveListKernel>& kernelsAlIn);
    virtual ~AffineFunctionSingle() = default;

    virtual void Compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel,
        ExecutionConfig const & execution) const override;

protected:
    const KernelMap<AffineActiveListKernel>& kernelsAl;
};

class AffineFunctionMulti : public AffineFunction
{
public:
    AffineFunctionMulti(const BaseAddress& input, const BaseAddress& output, const uint32_t vectorCount,
        std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
        std::unique_ptr<const Tensor> weightScaleFactors,
        const KernelMap<AffineKernel>& kernelsIn);
    virtual ~AffineFunctionMulti() = default;

    const std::unique_ptr<const Tensor> WeightScaleFactors; // AffineFunctionMulti1B

    static const FullCapabilitiesMap Capabilities;
};

}
