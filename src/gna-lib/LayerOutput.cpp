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

#include "LayerOutput.h"

#include "AffineLayerCapabilities.h"
#include "AffineLayers.h"
#include "AuxiliaryCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "Validator.h"


#include <algorithm>
#include <memory>

using namespace GNA;
using CnnCaps = GNA::ConvolutionalLayer2DCapabilities;

static const DataModeLimits _ModesGen0_9 =
{
    {GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
    Gna2StatusXnnErrorOutputBytes
};

static const DataModeLimits _ModesGen3 =
{
    {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_ACTIVATION_DISABLED},
    _ModesGen0_9.Error
};

const FullCapabilitiesMap LayerOutput::capabilities =
{
    {INTEL_AFFINE, {
        AffineLayerCapabilities::GetOperands(OutputOperandIndex).at(INTEL_AFFINE)
    }},
    {INTEL_AFFINE_DIAGONAL, {
        AffineLayerCapabilities::GetOperands(OutputOperandIndex).at(INTEL_AFFINE_DIAGONAL)
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        AffineLayerCapabilities::GetOperands(OutputOperandIndex).at(INTEL_AFFINE_MULTIBIAS)
    }},
    {INTEL_CONVOLUTIONAL, {
        ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex).at(INTEL_CONVOLUTIONAL)
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex).at(INTEL_CONVOLUTIONAL_2D)
    }},
    {INTEL_CONVOLUTIONAL_1D, {
        ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex).at(INTEL_CONVOLUTIONAL_1D)
    }},
    {INTEL_COPY, {
        AuxiliaryCapabilities::GetOperands(OutputOperandIndex).at(INTEL_COPY)
    }},
    {INTEL_DEINTERLEAVE, {
        AuxiliaryCapabilities::GetOperands(OutputOperandIndex).at(INTEL_DEINTERLEAVE)
    }},
    {INTEL_GMM, {
        GmmLayerCapabilities::GetOperands(OutputOperandIndex).at(INTEL_GMM)
    }},
    {INTEL_INTERLEAVE, {
        AuxiliaryCapabilities::GetOperands(OutputOperandIndex).at(INTEL_INTERLEAVE)
    }},
    {INTEL_RECURRENT, {
        AffineLayerCapabilities::GetOperands(OutputOperandIndex).at(INTEL_RECURRENT)
    }}
};

//TODO:3:Remove with API1
const FullCapabilitiesMap & LayerOutput::GetCapabilitiesLegacy()
{
    static FullCapabilitiesMap capabilitiesLegacy{capabilities};
    auto& cnnCaps = capabilitiesLegacy[INTEL_CONVOLUTIONAL_2D][Gna2DeviceGeneration3_0];
    const auto cnn2dLegacy = std::make_shared<TensorLimits>(TensorLimits{
            cnnCaps->Order,
            {{GNA_DIM_N, cnnCaps->Dimensions.at(GNA_DIM_N)},
             {GNA_DIM_H, {1, LayerCapabilities::InputElementCountMax * LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {1, LayerCapabilities::InputElementCountMax * LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_D, {1, LayerCapabilities::InputElementCountMax * LayerCapabilities::InputElementCountMax, 1, Gna2StatusXnnErrorOutputVolume}}},
            _ModesGen3});
    cnnCaps = cnn2dLegacy;
    return capabilitiesLegacy;
}

void * getScratchpadForOperation(const Gna2Operation &operation)
{
    if(operation.Type == Gna2OperationTypeTransposition ||
       operation.Type == Gna2OperationTypeCopy)
    {
        return nullptr;
    }
    //TODO:3:remove when final scratchpad impl provided
    return AffineBaseLayer::GetGlobal2MBScratchpad();
}

ApiShape LayerOutput::GetShape(const Gna2Operation & operation)
{
    ApiShape s{ operation.Operands[OutputOperandIndex]->Shape };
    if (operation.Type != Gna2OperationTypeConvolution ||
        s.NumberOfDimensions >= 4)
    {
        return s;
    }
    if (!CnnLayer::IsForced(operation))
    {
        ModelErrorHelper::ExpectEqual(s.NumberOfDimensions, 3, Gna2ItemTypeShapeNumberOfDimensions);
        s.NumberOfDimensions = 4;
        s.Dimensions[3] = s.Dimensions[2];
        s.Dimensions[2] = s.Dimensions[1];
        s.Dimensions[1] = 1;

    }
    return s;
}

//TODO:3:P1: Generalize instead addressing output at index 1
LayerOutput::LayerOutput(const Gna2Operation &operation, const LayerValidator& validatorIn)
try :
    Tensor{ Shape::Create(GetShape(operation), capabilities.GetOrder(validatorIn)),
        GetDataMode(*operation.Operands[OutputOperandIndex]), operation.Operands[OutputOperandIndex]->Data,
        Validator{ validatorIn, capabilities } },
    ScratchPad{Dimensions, Gna2DataTypeInt32, Gna2TensorModeDefault, getScratchpadForOperation(operation)}, //TODO:3:P1:Decide what to do with scratch pad in API2, disabled validation, as parameters are provided by library
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, OutputOperandIndex);
}

std::pair<uint32_t, uint32_t> LayerOutput::getGroupingAndElements(
      const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    switch (operation.Type)
    {
    case Gna2OperationTypeTransposition:
    {
        if (validatorIn.Operation == INTEL_INTERLEAVE)
        {
            return {Dimensions.at('W'), Dimensions.at('H')};
        }
        if (validatorIn.Operation == INTEL_DEINTERLEAVE)
        {
            return {Dimensions.at('H'), Dimensions.at('W')};
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    case Gna2OperationTypeGmm:
        return {Dimensions.at('W'), Dimensions.at('H')};
    default:
        return Tensor::getGroupingAndElements(operation, validatorIn);
    }
}

ModelValue LayerOutput::AsModelValue(char dimension) const
{
    auto mv = Component::AsModelValue(dimension);
    mv.SetOperand(OutputOperandIndex);
    return mv;
}
