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

#pragma once

#include "AccelerationDetector.h"
#include "ActivationFunction.h"
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

class RecurrentFunction : public Transform<RecurrentConfig, RecurrentKernel>
{
public:
    static std::unique_ptr<RecurrentFunction> Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);

    RecurrentFunction(const BaseTransformConfig<RecurrentKernel>& config,
        std::unique_ptr<const PwlCached> pwl,
        TransformOperation operation, uint32_t delay,
        std::unique_ptr<const WeightTensor> weights,
        std::unique_ptr<const BiasTensor> biases);

    virtual ~RecurrentFunction() = default;

    const BaseAddress CalculateFeedbackBuffer(const BaseAddress& outputBuffer) const;

    const ActivationFunction& GetActivationFunction() const;
    void SetActivationFunction(std::unique_ptr<ActivationFunction> activation);

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    std::unique_ptr<const WeightTensor> Weights;
    std::unique_ptr<const BiasTensor> Biases;

private:
    static const FullCapabilitiesMap outputCapabilities;
    static const std::map<Gna2OperationType, kernel_op> kernelOperationMap;

    virtual void UpdateConfigBuffers(
            std::unique_ptr<BaseConfig> configs[TransformOperationCount],
            const BufferMap& buffers) const override;

    static void ValidateActivation(const Gna2Tensor& activationTensor);
    void ValidateFeedbackDelay() const;

    const uint32_t FeedbackDelay;

    const std::unique_ptr<const PwlCached> pwl;
    const ActivationConfig activationConfig;
    std::unique_ptr<ActivationFunction> Activation;
};

}

