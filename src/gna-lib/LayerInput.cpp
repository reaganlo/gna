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

#include "LayerInput.h"

#include "AffineLayerCapabilities.h"
#include "AuxiliaryCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "GmmLayerCapabilities.h"
#include "Macros.h"
#include "ModelError.h"
#include "ModelWrapper.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>


using namespace GNA;

const FullCapabilitiesMap LayerInput::capabilities =
{
    GetOperationCaps<INTEL_AFFINE>(InputOperandIndex),
    GetOperationCaps<INTEL_AFFINE_DIAGONAL>(InputOperandIndex),
    GetOperationCaps<INTEL_AFFINE_MULTIBIAS>(InputOperandIndex),
    GetOperationCaps<INTEL_RECURRENT>(InputOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL>(InputOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL_2D>(InputOperandIndex),
    GetOperationCaps<INTEL_CONVOLUTIONAL_1D>(InputOperandIndex),
    GetOperationCaps<INTEL_GMM>(InputOperandIndex),
    GetOperationCaps<INTEL_COPY>(InputOperandIndex),
    GetOperationCaps<INTEL_INTERLEAVE>(InputOperandIndex),
    GetOperationCaps<INTEL_DEINTERLEAVE>(InputOperandIndex),
};

ApiShape LayerInput::GetShape(const Gna2Operation & operation)
{
    ApiShape shape{ operation.Operands[InputOperandIndex]->Shape };
    if (Gna2OperationTypeConvolution == operation.Type &&
        shape.NumberOfDimensions < 4 &&
        !CnnLayer::IsForced(operation))
    {
        ModelErrorHelper::ExpectEqual(shape.NumberOfDimensions, 2, Gna2ItemTypeShapeNumberOfDimensions);
        shape.Dimensions[2] = shape.Dimensions[1];
        shape.Dimensions[1] = 1;
        shape.Dimensions[3] = 1;
        shape.NumberOfDimensions = 4;
    }
    return shape;
}

LayerInput::LayerInput(const Gna2Operation& operation, const LayerValidator& validatorIn)
try :
    Tensor{ Shape::Create(GetShape(operation), capabilities.GetOrder(validatorIn)),
       GetDataMode(*operation.Operands[InputOperandIndex]), operation.Operands[InputOperandIndex]->Data,
       Validator{ validatorIn, capabilities } },
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, InputOperandIndex);
}

bool LayerInput::IsInputInterleave(const Gna2Tensor &apiTensor,
    const BaseValidator& validatorIn)
{
    auto layerValidator = LayerValidator{ validatorIn, INTEL_INTERLEAVE };
    try
    {
        Tensor{
           apiTensor, capabilities.GetOrder(layerValidator),
           Validator{ layerValidator, capabilities } };
        return true;
    }
    catch (const GnaException&)
    {
        return false;
    }
}

std::pair<uint32_t, uint32_t> LayerInput::getGroupingAndElements(
    const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    switch (operation.Type)
    {
    case Gna2OperationTypeTransposition:
    {
        if (validatorIn.Operation == INTEL_INTERLEAVE)
        {
            return { Dimensions.at('H'), Dimensions.at('W') };
        }
        if (validatorIn.Operation == INTEL_DEINTERLEAVE)
        {
            return { Dimensions.at('W'), Dimensions.at('H') };
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    default:
        return Tensor::getGroupingAndElements(operation, validatorIn);
    }
}

