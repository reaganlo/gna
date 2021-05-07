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

#include "AffineFunctions.h"

#include "AccelerationDetector.h"
#include "ActiveList.h"
#include "AffineLayerCapabilities.h"
#include "Bias.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GnaException.h"
#include "LayerConfiguration.h"
#include "OperationConfig.h"
#include "Shape.h"
#include "Validator.h"
#include "Weight.h"

#include "gna2-common-api.h"


#include <algorithm>
#include <cstdint>
#include <memory>

using namespace GNA;

const FullCapabilitiesMap AffineFunctionSingle::outputCapabilities =
{
    GetOperationCaps<INTEL_AFFINE>(OutputOperandIndex),
    GetOperationCaps<INTEL_AFFINE_DIAGONAL>(OutputOperandIndex),
    GetOperationCaps<INTEL_RECURRENT>(OutputOperandIndex),
};

const FullCapabilitiesMap AffineFunctionMulti::outputCapabilities =
{
    GetOperationCaps<INTEL_AFFINE_MULTIBIAS>(OutputOperandIndex),
};

const FullCapabilitiesMap AffineFunctionMulti::Capabilities =
{
    GetOperationCaps<INTEL_AFFINE_MULTIBIAS>(WeightScaleFactorOperandIndex),
};

// Could not split into separate methods for each component as multibias weight scaling is using bias' and weights; tensors...
std::unique_ptr<AffineFunction> AffineFunction::Create(
        const TransformFactoryConfig& config,
        const OperationConfig& operationConfig)
{
    if (operationConfig.BiasMode == Gna2BiasModeGrouping)
    {
        return createAffineMultiFunction(config, operationConfig);
    }

    return createAffineSingleFunction(config, operationConfig);
}

std::unique_ptr<AffineFunction> AffineFunction::createAffineSingleFunction(
    const TransformFactoryConfig& config, const OperationConfig& operationConfig)
{
    auto const kernelOperation = operationConfig.GetKernelOperation();
    auto weightTensor = operationConfig.WeightsTensor;
    auto biasTensor = operationConfig.BiasesTensor;
    auto weights = std::make_unique<const WeightTensor>(weightTensor, config.validator);
    auto biases = std::make_unique<const BiasTensor>(
            biasTensor, 0, Gna2BiasModeDefault, config.validator);
    auto const kernelMode = KernelMode { config.input->Mode, weights->Mode, biases->Mode };
    const auto& affineKernel = AccelerationDetector::GetKernelMap<AffineKernel>(
            kernelOperation, kernelMode);
    return std::make_unique<AffineFunctionSingle>(
            BaseTransformConfig<AffineKernel>{config, affineKernel},
            operationConfig.GetTransformOperation(),
            std::move(weights), std::move(biases));
}

std::unique_ptr<AffineFunction> AffineFunction::createAffineMultiFunction(
    const TransformFactoryConfig& config, const OperationConfig& operationConfig)
{
    std::unique_ptr<const Tensor> weightScales;
    auto weightTensor = operationConfig.WeightsTensor;
    auto biasTensor = operationConfig.BiasesTensor;
    auto biasVectorIndex = operationConfig.BiasVectorIndex;
    auto weights = std::make_unique<const WeightTensor>(weightTensor, config.validator);
    auto biases = std::make_unique<const BiasTensor>(biasTensor, biasVectorIndex,
            Gna2BiasModeGrouping, config.validator);

    if (operationConfig.WeightScalesTensor.Mode != Gna2TensorModeDisabled)
    {
        const std::function<void()> command = [&]()
        {
            weightScales = std::make_unique<const Tensor>(operationConfig.WeightScalesTensor,
                Validator{ config.validator, AffineFunctionMulti::Capabilities });
            ModelErrorHelper::ExpectNotNull(*weightScales);
        };
        ModelErrorHelper::ExecuteForModelItem(command, WeightScaleFactorOperandIndex);
    }

    auto const kernelOperation = KERNEL_AFFINE_MULTIBIAS;
    auto const kernelMode = KernelMode { config.input->Mode, weights->Mode, biases->Mode };
    auto& affineKernel = AccelerationDetector::GetKernelMap<AffineKernel>(
            kernelOperation, kernelMode);
    return std::make_unique<AffineFunctionMulti>(BaseTransformConfig<AffineKernel>{config, affineKernel},
            operationConfig.GetTransformOperation(),
            std::move(weights), std::move(biases),
            std::move(weightScales));
}

AffineFunction::AffineFunction(const BaseTransformConfig<AffineKernel>& config,
    TransformOperation transform,
    std::unique_ptr<const WeightTensor> weights,
    std::unique_ptr<const BiasTensor> biases) :
    Transform{transform, &config.kernels, config.input},
    Weights{ std::move(weights) },
    Biases{ std::move(biases) }
{
}

Tensor const& AffineFunction::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case WeightOperandIndex:
    {
        return GetOperandIfExistOrThrow(Weights);
    }
    case BiasOperandIndex:
    {
        return GetOperandIfExistOrThrow(Biases);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}

AffineFunctionSingle::AffineFunctionSingle(
    BaseTransformConfig<AffineKernel> config, TransformOperation transform,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases)
    : AffineFunction(config, transform, std::move(weights), std::move(biases)),
    kernelsAl(AccelerationDetector::GetKernelMap<AffineActiveListKernel>(
        KERNEL_AFFINE_AL, KernelMode { config.input->Mode, Weights->Mode, Biases->Mode }))
{
    AffineConfig kernelAffineConfig = { config.output->Dimensions.at('H'),
        config.output->Dimensions.at('W'), config.input->Dimensions.at('H'),
        config.input->Buffer, config.output->Buffer, *Weights,
        *Biases, nullptr, 0, Biases->Mode.Size};

    Output = std::make_unique<Tensor>(
        Shape{GNA_TENSOR_HW, config.output->Dimensions.at('H'), config.output->Dimensions.at('W')},
        config.output->Mode, config.outputBuffer,
        Validator{config.validator, outputCapabilities});

    hiddenConfig = std::make_unique<KernelConfig<AffineConfig>>(kernelAffineConfig,
            BaseConfig { Input->Buffer, Output->Buffer });
}

void AffineFunctionSingle::ValidateActiveList(ActiveList const& activeList) const
{
    Expect::InRange(activeList.IndicesCount,
        1u, Output->at(GNA_DIM_H), Gna2StatusActiveListIndicesInvalid);
    // Only Int32 is supported with active list
    Expect::InSet(Biases->Mode.Type,
        { Gna2DataTypeInt32, Gna2DataTypeCompoundBias },
        Gna2StatusModelConfigurationInvalid);
}

void AffineFunctionSingle::Compute(AccelerationMode accel,
    LayerConfiguration const * layerConfiguration, ExecutionConfig const & execution) const
{
    auto executionConfig = createExecutionConfig(layerConfiguration, execution);
    try
    {
        if (layerConfiguration != nullptr && layerConfiguration->ActList)
        {
            kernelsAl.at(accel)(executionConfig.get(), AffineConfigAl{
                                layerConfiguration->ActList->Indices,
                                layerConfiguration->ActList->IndicesCount});
        }
        else
        {
            kernels->at(accel)(executionConfig.get());
        }
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusNotImplemented);
    }
}

AffineFunctionMulti::AffineFunctionMulti(BaseTransformConfig<AffineKernel> config,
    TransformOperation transform,
    std::unique_ptr<const WeightTensor> weights, std::unique_ptr<const BiasTensor> biases,
    std::unique_ptr<const Tensor> weightScaleFactors) :
    AffineFunction(config, transform, std::move(weights), std::move(biases)),
    WeightScaleFactors{ std::move(weightScaleFactors) }
{
    AffineConfig kernelAffineConfig = { config.output->Dimensions.at('H'),
        config.input->Dimensions.at('W'), config.input->Dimensions.at('H'),
        config.input->Buffer, config.output->Buffer, *Weights,
        (WeightScaleFactors ? static_cast<const void*>(*WeightScaleFactors) : nullptr),
        *Biases, Biases->Dimensions.at('W'), Biases->Mode.Size };

    Output = std::make_unique<Tensor>(
        Shape{GNA_TENSOR_HW, config.output->Dimensions.at('H'), config.output->Dimensions.at('W')},
        config.output->Mode, config.outputBuffer,
        Validator{config.validator, outputCapabilities});

    hiddenConfig = std::make_unique<KernelConfig<AffineConfig>>(kernelAffineConfig,
            BaseConfig { Input->Buffer, Output->Buffer });
}

Tensor const& AffineFunctionMulti::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case WeightScaleFactorOperandIndex:
    {
        return GetOperandIfExistOrThrow(WeightScaleFactors);
    }
    default:
        return AffineFunction::GetOperand(operandIndex);
    }
}
