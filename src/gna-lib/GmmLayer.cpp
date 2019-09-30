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

#include "GmmLayer.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "ActiveList.h"
#include "Capabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"
#include "gna-api.h"

#include <algorithm>
#include <memory>

using namespace GNA;

void GmmOperation::VerifyHas1BInputAnd2BWeight()
{}

Tensor const & GmmOperation::GetOperand(uint32_t operandIndex) const
{
    // TODO:3:replace with generic solution when all layers are transforms
    switch (operandIndex)
    {
    case GmmMeanOperandIndex://[[fallthrough]] also same value for GmmInterleavedOperandIndex
    case GmmInverseCovarianceOperandIndex://[[fallthrough]]
    case GmmGaussianConstantOperandIndex:
    {
        return getTransformOperand(GmmTransform, operandIndex);
    }
    default:
        return Layer::GetOperand(operandIndex);
    }
}

DataConfig GmmOperation::GetDataMode() const
{
    return reinterpret_cast<GmmFunction const *>(inputTransform)->GetDataMode();
}

std::unique_ptr<GmmFunction> GmmFunction::Create(const TransformFactoryConfig& config,
    const OperationConfig& operation)
{
    auto const isInterleaved = operation.Operation->NumberOfOperands == 3;
    auto varMode = GNA_UINT8;
    std::unique_ptr<const WeightTensor> means;
    std::unique_ptr<const WeightTensor> inverseCovariances;
    std::unique_ptr<const BiasTensor> gaussianConstants;

    if (!isInterleaved)
    {
        auto const meanTensor = operation.GetOperand(GmmMeanOperandIndex);
        means = std::make_unique<const WeightTensor>(meanTensor, config.validator);

        auto const inverseCovariancesTensor = operation.GetOperand(GmmInverseCovarianceOperandIndex);
        inverseCovariances = std::make_unique<const WeightTensor>(inverseCovariancesTensor, config.validator);

        auto const gaussianConstantsTensor = operation.GetOperand(GmmGaussianConstantOperandIndex);
        gaussianConstants = std::make_unique<const BiasTensor>(
            gaussianConstantsTensor, 0, Gna2BiasModeDefault, config.validator);

        varMode = inverseCovariances->Mode.Value;
    }
    else
    {
        auto const meanTensor = operation.GetOperand(GmmInterleavedOperandIndex);
        means = std::make_unique<const WeightTensor>(meanTensor, config.validator);

        varMode = means->Mode.Value;
    }

    auto const kernelMode = KernelMode{ GNA_UINT8, varMode, GNA_UINT32 };
    const auto& gmmKernels = AccelerationDetector::GetKernelMap<GmmMaxMix>(KERNEL_GMM, kernelMode);
    const auto& gmmKernelsAl = AccelerationDetector::GetKernelMap<GmmMaxMixActiveList>(KERNEL_GMM_AL, kernelMode);
    auto const maximumScore = operation.GetParameterAs<uint32_t>(0);

    if (!isInterleaved)
    {
        return std::make_unique<GmmFunctionFlat>(
            BaseTransformConfig<GmmMaxMix>{config, gmmKernels},
            std::move(means), std::move(inverseCovariances), std::move(gaussianConstants),
            maximumScore, gmmKernelsAl);
    }
    else
    {
        return std::make_unique<GmmFunctionInterleaved>(
            BaseTransformConfig<GmmMaxMix>{config, gmmKernels},
            std::move(means),
            maximumScore, gmmKernelsAl);
    }
}

Tensor const& GmmFunction::GetOperand(uint32_t operandIndex) const
{
    // TODO:3:replace with generic solution when all layers are transforms
    switch (operandIndex)
    {
    case GmmMeanOperandIndex:
    {
        return GetOperandIfExistOrThrow(Means);
    }
    case GmmInverseCovarianceOperandIndex:
    {
        return GetOperandIfExistOrThrow(InverseCovariances);
    }
    case GmmGaussianConstantOperandIndex:
    {
        return GetOperandIfExistOrThrow(GaussianConstants);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}

void GmmFunction::ValidateActiveList(ActiveList const & activeList) const
{
    Expect::InRange(activeList.IndicesCount,
        ui32_1, Means->at(GNA_DIM_H), Gna2StatusActiveListIndicesInvalid);
}

GmmFunction::GmmFunction(const BaseTransformConfig<GmmMaxMix>& config,
    std::unique_ptr<const WeightTensor> means,
    std::unique_ptr<const WeightTensor> inverseCovariances,
    std::unique_ptr<const BiasTensor> gaussianConstants,
    uint32_t const maximumScore,
    const KernelMap<GmmMaxMixActiveList>& gmmKernelsAl) :
    TransformAl{ GmmTransform, &config.kernels, &gmmKernelsAl, config.input },
    Means{ std::move(means) },
    InverseCovariances{ std::move(inverseCovariances) },
    GaussianConstants{ std::move(gaussianConstants) },
    MaximumScore{ maximumScore }
{
    Expect::NotNull(Means);

    MeanBuffer = Means->Buffer;
    StateCount = Means->at(GNA_DIM_H);

    auto const mixCount = Means->at(GNA_DIM_W);
    auto const inElementCount = Input->at(GNA_DIM_W);
    InverseCovarianceSize = InverseCovariances ? InverseCovariances->Mode.Size : Means->Mode.Size;

    MeanSetOffsetSize = mixCount * inElementCount * GMM_MEAN_VALUE_SIZE;
    VarSetOffsetSize = mixCount * inElementCount * InverseCovarianceSize;
    GaussConstSetOffsetSize = RoundUp(mixCount, 2) * GMM_CONSTANTS_SIZE;

    Expect::InRange(MeanSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_MEAN_VALUE_SIZE, Gna2StatusGmmBadMeanSetoff);
    Expect::MultiplicityOf(MeanSetOffsetSize, GMM_MEM_ALIGNMENT);
    Expect::InRange(VarSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_FV_ELEMENT_COUNT_MAX * GMM_COVARIANCE_SIZE_MAX, Gna2StatusGmmBadVarSetoff);
    Expect::MultiplicityOf(VarSetOffsetSize, GMM_MEM_ALIGNMENT);
    Expect::InRange(GaussConstSetOffsetSize, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF,
        GMM_MIXTURE_COMP_COUNT_MAX * GMM_CONSTANTS_SIZE, Gna2StatusGmmBadGconstOffset);
    Expect::MultiplicityOf(GaussConstSetOffsetSize, GMM_MEM_ALIGNMENT);

    Output = std::make_unique<Tensor>(config.output->Dimensions, config.output->Mode,
        config.output->Buffer, Validator{ config.validator, getOutputCapabilities() });
}

void GmmFunction::InitHiddenConfig()
{
    auto const gmmConfig = GmmConfig{ Input->at(GNA_DIM_H),
       Input->at(GNA_DIM_W),
       Means->at(GNA_DIM_W),
       MeanSetOffsetSize,
       VarSetOffsetSize,
       GaussConstSetOffsetSize,
       MaximumScore,
       StateCount,
       MeanBuffer,
       InverseCovarianceBuffer,
       GaussianConstantBuffer,
    };
    hiddenConfig = std::make_unique<KernelConfig<GmmConfig>>(gmmConfig,
        BaseConfig{ Input->Buffer, Output->Buffer });
}

const FullCapabilitiesMap& GmmFunction::getOutputCapabilities()
{
    // TODO:3:KJ: move to new GmmOperationCapabilities
    static const FullCapabilitiesMap capabilities =
    {
     {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW}, // H - GMM States, W - grouping
            {{GNA_DIM_W, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            { { GNA_UINT32, GNA_DATA_ACTIVATION_DISABLED }, Gna2StatusXnnErrorOutputBytes }})}
    }},
    };
    return capabilities;
}

GmmFunctionFlat::GmmFunctionFlat(
    const BaseTransformConfig<void(*)(ExecutionKernelConfig<GmmConfig> const* const)>& config,
    std::unique_ptr<const WeightTensor> means, std::unique_ptr<const WeightTensor> inverseCovariances,
    std::unique_ptr<const BiasTensor> gaussianConstants, uint32_t const maximumScore,
    const KernelMap<void(*)(ExecutionKernelConfig<GmmConfig> const* const, AffineConfigAl al)>&kernelsAlIn) :
    GmmFunction{ config, std::move(means), std::move(inverseCovariances), std::move(gaussianConstants), maximumScore, kernelsAlIn }
{
    Expect::NotNull(GaussianConstants);
    Expect::NotNull(InverseCovariances);

    // TODO:3:KJ:move this to mode capabilities of Means->inherit WeightTensor
    Expect::Equal(Means->Mode.Type, Gna2DataTypeUint8, Gna2StatusDataModeInvalid);

    InverseCovarianceBuffer = InverseCovariances->Buffer;
    GaussianConstantBuffer = GaussianConstants->Buffer;

    config.validator.ValidateBufferIfSet(GaussianConstants->Buffer, StateCount * GaussConstSetOffsetSize);
    config.validator.ValidateBufferIfSet(Means->Buffer, StateCount * MeanSetOffsetSize);
    config.validator.ValidateBufferIfSet(InverseCovariances->Buffer, StateCount * VarSetOffsetSize);

    InitHiddenConfig();
}

DataConfig GmmFunctionFlat::GetDataMode() const
{
    return DataConfig(Input->Mode, InverseCovariances->Mode, GNA_UINT32, Output->Mode);
}

GmmFunctionInterleaved::GmmFunctionInterleaved(
    const BaseTransformConfig<void(*)(ExecutionKernelConfig<GmmConfig> const* const)>& config,
    std::unique_ptr<const WeightTensor> interleavedData, uint32_t const maximumScore,
    const KernelMap<GmmMaxMixActiveList>& kernelsAlIn) :
    GmmFunction{ config, std::move(interleavedData), nullptr, nullptr,
        maximumScore, kernelsAlIn }
{
    InverseCovarianceBuffer = Means->Buffer + MeanSetOffsetSize;
    GaussianConstantBuffer = InverseCovarianceBuffer + VarSetOffsetSize;

    auto const interleavedSetOffsetSize = MeanSetOffsetSize + VarSetOffsetSize + GaussConstSetOffsetSize;;
    MeanSetOffsetSize = interleavedSetOffsetSize;
    VarSetOffsetSize = interleavedSetOffsetSize;
    GaussConstSetOffsetSize = interleavedSetOffsetSize;

    config.validator.ValidateBufferIfSet(Means->Buffer, StateCount * MeanSetOffsetSize);

    InitHiddenConfig();
}

DataConfig GmmFunctionInterleaved::GetDataMode() const
{
    return DataConfig(Input->Mode, Means->Mode, GNA_UINT32, Output->Mode);
}
