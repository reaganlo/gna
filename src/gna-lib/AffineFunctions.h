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

#include "AccelerationDetector.h"
#include "Address.h"
#include "Bias.h"
#include "KernelArguments.h"
#include "Tensor.h"
#include "Transform.h"
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
class OperationConfig;

struct LayerConfiguration;

// AffineFunction interface
struct AffineFunction : public Transform<AffineConfig, AffineKernel>
{
public:
    virtual ~AffineFunction() = default;

    static std::unique_ptr<AffineFunction> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);

    Tensor const& GetOperand(uint32_t operandIndex) const override;

    std::unique_ptr<const WeightTensor> Weights;
    std::unique_ptr<const BiasTensor> Biases;

protected:
    AffineFunction(const BaseTransformConfig<AffineKernel>& config,
        TransformOperation operation,
        std::unique_ptr<const WeightTensor> weights,
        std::unique_ptr<const BiasTensor> biases);

    static const ShapeLimits outputDimensionsLimits;

    static const DataModeLimits outputModeLimits_0_9;

    static const TensorLimits outputLimits_0_9;

    static const DataModeLimits outputModeLimits_3;

    static const TensorLimits outputLimits_3;

private:
    static const std::map<Gna2OperationType, kernel_op> kernelOperationMap;
    static std::unique_ptr<AffineFunction> createAffineSingleFunction(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);
    static std::unique_ptr<AffineFunction> createAffineMultiFunction(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);
};

class AffineFunctionSingle : public AffineFunction
{
public:
    AffineFunctionSingle(BaseTransformConfig<AffineKernel> config,
        TransformOperation transform,
        std::unique_ptr<const WeightTensor> weights,
        std::unique_ptr<const BiasTensor> biases);

    virtual ~AffineFunctionSingle() = default;

    void ValidateActiveList(ActiveList const & activeList) const override;

    void Compute(AccelerationMode accel, LayerConfiguration const* layerConfiguration,
                 ExecutionConfig const& execution) const override;

private:
    static const FullCapabilitiesMap outputCapabilities;

    const KernelMap<AffineActiveListKernel>& kernelsAl;
};

class AffineFunctionMulti : public AffineFunction
{
public:
    AffineFunctionMulti(BaseTransformConfig<AffineKernel> config,
        TransformOperation transform,
        std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
        std::unique_ptr<const Tensor> weightScaleFactors);
    virtual ~AffineFunctionMulti() = default;

    Tensor const& GetOperand(uint32_t operandIndex) const override;

    const std::unique_ptr<const Tensor> WeightScaleFactors; // AffineFunctionMulti1B

    static const FullCapabilitiesMap Capabilities;

private:
    static const FullCapabilitiesMap outputCapabilities;
};
}
