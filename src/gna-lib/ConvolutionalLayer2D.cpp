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

#include "ConvolutionalLayer2D.h"

#include "ActivationFunction.h"
#include "Address.h"
#include "ConvolutionalFunctions2D.h"
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

#include "gna-api-types-xnn.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace GNA;

void ConvolutionalLayer2D::Init()
{
    Expect::One(Input.at(GNA_DIM_N), Gna2StatusXnnErrorGrouping);
    Expect::One(Output.at(GNA_DIM_N), Gna2StatusXnnErrorGrouping);
    Expect::Equal(Output.Size, GetOutputTransform()->Output->Size, Gna2StatusXnnErrorOutputVolume);

    // performed for layer size validation
    auto uArchConfig = HardwareLayerCnn2D::CalculateUArchConfig(
        validator->HwCapabilities.GetDeviceVersion(),
        Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D),
        Transforms.Get<PoolingFunction2D>(PoolingTransform2D),
        GetOutputTransform()->Output->Mode);

    Expect::True(uArchConfig.Valid, Gna2StatusModelConfigurationInvalid);

    Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(nullptr, accel, executionConfig); };

    Layer::Compute = [this](LayerConfiguration &layerConfiguration,
        AccelerationMode accel,
        ExecutionConfig const & executionConfig)
    {this->compute(&layerConfiguration, accel, executionConfig); };
}

Tensor const & ConvolutionalLayer2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case 2://[[fallthrough]]
    case 3:
    {
        return getTransformOperand(ConvolutionalTransform2D, operandIndex);
    }
    case 4:
    {
        return getTransformOperand(ActivationTransform, 2);// TODO:3:Intentional literal, replace with generic solution when all layers are transforms
    }
    default:
        return Layer::GetOperand(operandIndex);
    }
}

bool ConvolutionalLayer2D::IsSupported(const Gna2Operation & operation)
{
    return 4 == operation.Operands[0]->Shape.NumberOfDimensions;
}

DataConfig ConvolutionalLayer2D::GetDataMode() const
{
    auto& convolutionTransform = *Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D);
    const auto filterMode = convolutionTransform.Filters->Mode.Value;
    const auto biasMode = convolutionTransform.Biases->Mode.Value;
    return DataConfig(Input.Mode.Value, filterMode, biasMode, Output.Mode.Value);
}
