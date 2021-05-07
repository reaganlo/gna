/*
 INTEL CONFIDENTIAL
 Copyright 2019-2020 Intel Corporation.

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

#include "ConvolutionalLayer2D.h"

#include "ActivationFunction.h"
#include "Address.h"
#include "ConvolutionalFunctions2D.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "KernelArguments.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "PoolingFunctions2D.h"
#include "Tensor.h"
#include "Transform.h"
#include "TransformMap.h"
#include "Validator.h"


#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace GNA;

void ConvolutionalLayer2D::Init()
{
    auto const is1D = GetInputTransform().Is1D() &&
        (Transforms.GetOptional<PoolingFunction2D>(PoolingTransform2D) == nullptr ||
         GetOutputTransform().Is1D());
    if (is1D)
    {
        auto const& capsMapIn = ConvolutionalLayer2DCapabilities::GetOperands(InputOperandIndex);
        Input.Validate(capsMapIn, INTEL_CONVOLUTIONAL_1D);

        auto const& capsMapOut = ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex);
        Output.Validate(capsMapOut, INTEL_CONVOLUTIONAL_1D);
    }
    else if(Gna2DeviceGeneration3_5 < validator->HwCapabilities.GetDeviceGeneration())
    {
        auto const precision = Output.Mode.Size;
        if (precision < 4)
        {
            auto const& filters = getTransformOperand(ConvolutionalTransform2D, FilterOperandIndex);
            auto const filterCount = filters.at(GNA_DIM_N);
            if (filterCount > 2)
            {
                Expect::MultiplicityOf(filterCount, 4 / precision,
                    Gna2StatusCnnErrorConvFltCount);
            }
        }
    }
    Validate3_0ExtraLimits(is1D);

    Expect::One(Output.at(GNA_DIM_N), Gna2StatusXnnErrorGrouping);
    Expect::Equal(Output.Size, GetOutputTransform().Output->Size, Gna2StatusXnnErrorOutputVolume);
    auto const & convolutionTransform = Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D);
    const auto filterMode = convolutionTransform.Filters->Mode;
    const auto biasMode = convolutionTransform.Biases->Mode;
    auto const activation = Transforms.GetOptional<ActivationFunction>(ActivationTransform);
    dataConfig = { Input.Mode, filterMode, biasMode, Output.Mode, activation == nullptr };
}

void ConvolutionalLayer2D::Validate3_0ExtraLimits(bool is1D) const
{
    if (!is1D && Gna2DeviceGeneration3_0 == validator->HwCapabilities.GetDeviceGeneration())
    {
        auto const activation = Transforms.GetOptional(ActivationTransform);
        auto const pooling = Transforms.GetOptional<PoolingFunction2D>(PoolingTransform2D);
        auto const & filter = GetInputTransform().GetOperand(FilterOperandIndex);
        Expect::Equal(Input.Mode, filter.Mode, Gna2StatusXnnErrorConvFltBytes);
        Expect::True(nullptr == activation || Output.Mode.Type == Input.Mode.Type, Gna2StatusXnnErrorOutputBytes);
        if(pooling)
        {
            if (activation && Output.Mode.Type == Gna2DataTypeInt8)
            {
                Expect::True(pooling->Mode != KernelPoolingModeMax, Gna2StatusXnnErrorLyrCfg);
            }
            pooling->Window->Dimensions.ExpectSquare();
        }
        if (2 == filter.Mode.Size)
        {
            if (filter.at(GNA_DIM_W) > 1)
            {
                Expect::InRange(Input.at(GNA_DIM_D), 120u, Gna2StatusXnnErrorInputVolume);
                if (!filter.Dimensions.IsSquare())
                {
                    Expect::One(filter.at(GNA_DIM_H), Gna2StatusXnnErrorWeightVolume);
                }
            }
        }
        else
        {
            if (filter.at(GNA_DIM_W) > 2)
            {
                Expect::InRange(Input.at(GNA_DIM_D), 240u, Gna2StatusXnnErrorInputVolume);
            }
            if (!filter.Dimensions.IsSquare())
            {
                Expect::True(filter.at(GNA_DIM_W) == 1 || filter.at(GNA_DIM_H) == 1, Gna2StatusXnnErrorWeightVolume);
            }
        }
    }
}

Tensor const & ConvolutionalLayer2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        if (Transforms.GetOptional(ActivationTransform))
        {
            return Output.ScratchPad;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case FilterOperandIndex://[[fallthrough]]
    case BiasOperandIndex:
    {
        return getTransformOperand(ConvolutionalTransform2D, operandIndex);
    }
    case PwlOperandIndex:
    {
        return getTransformOperand(ActivationTransform, 2);// TODO:3:Intentional literal, replace with generic solution when all layers are transforms
    }
    case SoftwareScratchpadOperandIndex:
    {
        if (Transforms.size() > 1)
        {
            return GetInputTransform().GetOperand(OutputOperandIndex);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    default:
        return Layer::GetOperand(operandIndex);
    }
}

std::unique_ptr<const Component> ConvolutionalLayer2D::CreateComponentFromParameter(const Shape& shape,
    const LayerValidator& validatorIn, const uint32_t parameterIndex)
{
    std::unique_ptr<const Component> parameter;
    const std::function<void()> command = [&]()
    {
        parameter = OperationConfig::CreateCnnComponent(shape,
            validatorIn, ConvolutionalLayer2DCapabilities::GetParameters(parameterIndex));
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, static_cast<int32_t>(parameterIndex));
    return parameter;
}
