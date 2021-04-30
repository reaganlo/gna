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

#include "ActivationFunction.h"

#include "ActivationHelper.h"
#include "AccelerationDetector.h"
#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GnaException.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "Shape.h"
#include "Validator.h"

#include "gna2-common-api.h"


#include <algorithm>
#include <memory>

using namespace GNA;

static const auto pwlLimit = std::make_shared<TensorLimits>(
    TensorLimits{
        {GNA_TENSOR_H},    // W - #inputs, H - #outputs
        {{GNA_DIM_H, {ActivationFunction::ActivationFunctionSegmentCountMin, ActivationFunction::ActivationFunctionSegmentCountMax, 1, Gna2StatusXnnErrorPwlSegments}}},
        {{ Gna2DataTypePwlSegment }, Gna2StatusXnnErrorOutputBytes}});

const FullCapabilitiesMap ActivationFunction::capabilities =
{
    {INTEL_AFFINE,{
        {Gna2DeviceGeneration0_9, pwlLimit},
    }},
    {INTEL_AFFINE_DIAGONAL,{
        {Gna2DeviceGeneration0_9, pwlLimit},
    }},
    {INTEL_AFFINE_MULTIBIAS,{
        {Gna2DeviceGeneration2_0, pwlLimit},
    }},
    {INTEL_CONVOLUTIONAL,{
        {Gna2DeviceGeneration1_0, pwlLimit},
    }},
    {INTEL_CONVOLUTIONAL_2D,{
        {Gna2DeviceGeneration3_0, pwlLimit}
    }},
    {INTEL_RECURRENT,{
        {Gna2DeviceGeneration0_9, pwlLimit},
    }}
};

const FullCapabilitiesMap ActivationFunction::outputCapabilities =
{
    GetOperationCaps<INTEL_AFFINE>(OutputOperandIndex),
    GetOperationCaps<INTEL_AFFINE_DIAGONAL>(OutputOperandIndex),
    GetOperationCaps<INTEL_AFFINE_MULTIBIAS>(OutputOperandIndex),
    GetOperationCaps<INTEL_RECURRENT>(OutputOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL>(OutputOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL_2D>(OutputOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL_1D>(OutputOperandIndex),
};

std::unique_ptr<ActivationFunction> ActivationFunction::Create(const TransformFactoryConfig& config)
{
    if (config.IsActivationNotSupported())
    {
        return std::unique_ptr<ActivationFunction>(nullptr);
    }
    const auto mandatory = config.HasMandatoryActivation(); // TODO:3: use CAPS to determine

    const Gna2Tensor activation = config.GetActivation();

    if (mandatory || activation.Mode != Gna2TensorModeDisabled)
    {
        try
        {
            ActivationHelper::ExpectProper(activation);
            auto pwlFunction = std::make_unique<Tensor>(
                Shape(GNA_TENSOR_H, activation.Shape.Dimensions[0]),
                Gna2DataTypePwlSegment, activation.Data, Validator{ config.validator, capabilities });
            return std::make_unique<ActivationFunction>(
                BaseTransformConfig<ActivationKernel>{config,
                AccelerationDetector::GetKernelMap<ActivationKernel>(KERNEL_PWL)}, config.outputMode,
                std::move(pwlFunction));
        }
        catch (GnaException& e)
        {
            ModelErrorHelper::SetOperandIndexRethrow(e, PwlOperandIndex);
        }
    }
    return std::unique_ptr<ActivationFunction>(nullptr);
}

void ActivationFunction::UpdateActiveOutputCount(
    std::unique_ptr<BaseConfig> configs[TransformOperationCount], uint32_t outputCount) const
{
    auto config = GetConfig(configs);
    config->Transform.ElementCount = outputCount;
}


PwlCached ActivationFunction::createPwlCached(uint32_t elementSize,
    PwlSegment const * const segmentsIn, uint32_t segmentCountIn)
{
    try
    {
        return PwlCached(elementSize, segmentsIn, segmentCountIn);
    }
    catch (const std::runtime_error&)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
}

ActivationFunction::ActivationFunction(const BaseTransformConfig<ActivationKernel>& config,
    const DataMode& mode, std::unique_ptr<Tensor> pwl) :
    Transform{ ActivationTransform, &config.kernels, config.input },
    Segments{ std::move(pwl) },
    Pwl{ createPwlCached(config.outputMode.Size, Segments->Buffer, Segments->Count) }
{
    const auto validator = Validator{ config.validator, outputCapabilities };
    Output = std::make_unique<Tensor>(config.input->Dimensions, mode, config.outputBuffer,
        validator);

    hiddenConfig = std::make_unique<KernelConfig<ActivationConfig>>(
        ActivationConfig{ Output->Count, &Pwl }, BaseConfig{ Input->Buffer, config.outputBuffer });
}

Tensor const & ActivationFunction::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case 2:
    {
        return GetOperandIfExistOrThrow(Segments);
    }
    default:
        return Transform::GetOperand(operandIndex);
    }
}

void ActivationFunction::ValidateActiveList(ActiveList const& activeList) const
{
    Expect::InRange(activeList.IndicesCount,
        1u, Output->at(GNA_DIM_H), Gna2StatusActiveListIndicesInvalid);
}
