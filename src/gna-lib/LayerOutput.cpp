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

#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "DataMode.h"
#include "Expect.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include "gna-api.h"
#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <memory>

using namespace GNA;

static const std::vector<uint32_t> _Multipliers =
{ 2 * XNN_N_IN_ELEMS_MPLY, 1 * XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MPLY / 2};

static const ShapeLimits _FlatLimits =
{
    {GNA_DIM_H, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
    {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}
};

static const ShapeLimits _InterleaveLimits =
{
    {GNA_DIM_H, _FlatLimits.at(GNA_DIM_W)},
    {GNA_DIM_W, _FlatLimits.at(GNA_DIM_H)}
};

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

static const DataModeLimits _ModesCopy =
{
    {GNA_INT16},
    _ModesGen0_9.Error
};

static const DataModeLimits _ModesCopyGen3 =
{
    {GNA_INT8, GNA_INT16},
    _ModesGen0_9.Error
};

static const TensorLimits _InterleaveTensorLimitsGen0_9 =
{
    {GNA_TENSOR_HW},
    _InterleaveLimits,
    _ModesGen0_9
};

static const TensorLimits _FlatTensorLimitsGen0_9 =
{
    _InterleaveTensorLimitsGen0_9.Order,
    _FlatLimits,
    _ModesGen0_9
};

static const TensorLimits _InterleaveTensorLimitsGen3 =
{
    _InterleaveTensorLimitsGen0_9.Order,
    _InterleaveTensorLimitsGen0_9.Dimensions,
    _ModesGen3
};

static const TensorLimits _FlatTensorLimitsGen3 =
{
     _FlatTensorLimitsGen0_9.Order,
    _FlatTensorLimitsGen0_9.Dimensions,
    _ModesGen3
};

const FullCapabilitiesMap LayerOutput::capabilities =
{
    {INTEL_AFFINE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_DIAGONAL, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        {GNA_2_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
    {INTEL_CONVOLUTIONAL, {
        ConvolutionalLayer2DCapabilities::GetLegacyOperands(OutputOperandIndex)
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex)
    }},
    {INTEL_COPY, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            _FlatLimits,
            _ModesCopy})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, COPY_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
            {GNA_DIM_W, _FlatLimits.at(GNA_DIM_W)}},
            _ModesCopyGen3})}
    }},
    {INTEL_DEINTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
    {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HN}, // H - GMM States, N - grouping
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            { { GNA_INT32, GNA_DATA_ACTIVATION_DISABLED }, Gna2StatusXnnErrorOutputBytes }})}
    }},
    {INTEL_INTERLEAVE, {
        { GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9) },
        { GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3) }
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, Gna2StatusXnnErrorOutputVolume}}}, // must be multiple 32 to keep 64B output buffer alignment
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {RNN_N_OUT_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, RNN_N_OUT_ELEMS_MPLY, Gna2StatusXnnErrorOutputVolume}}}, // must be multiple 32 to keep 64B output buffer alignment
            _ModesGen3})}
    }}
};

//TODO:3:Remove with API1
const FullCapabilitiesMap & LayerOutput::GetCapabilitiesLegacy()
{
    static FullCapabilitiesMap capabilitiesLegacy{capabilities};
    auto& cnnCaps = capabilitiesLegacy[INTEL_CONVOLUTIONAL_2D][GNA_3_0];
    const auto cnn2dLegacy = std::make_shared<TensorLimits>(TensorLimits{
            cnnCaps->Order,
            {{GNA_DIM_N, cnnCaps->Dimensions.at(GNA_DIM_N)},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX * XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX * XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX * XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
            _ModesGen3});
    cnnCaps = cnn2dLegacy;
    return capabilitiesLegacy;
}

Shape LayerOutput::ConvertInCaseOfNewApiOrder(gna_tensor_order order, const uint32_t nOutputColumns, const uint32_t nOutputRows)
{
    if (order == GNA_TENSOR_NHWD)
    {
        return Shape{ GNA_TENSOR_NHWD, nOutputRows, nOutputColumns, 1u, 1u };
    }
    if (order == GNA_TENSOR_HW)
    {
        return Shape(order, nOutputRows, nOutputColumns);
    }
    return Shape(order, nOutputColumns, nOutputRows);
}

LayerOutput::LayerOutput(const nn_layer& layer, const LayerValidator& validatorIn) :
    Tensor{
        ConvertInCaseOfNewApiOrder( capabilities.GetOrder(validatorIn), layer.nOutputColumns, layer.nOutputRows ),
        layer.nBytesPerOutput, layer.pOutputs,
        Validator{ validatorIn, GetCapabilitiesLegacy() } },
        ScratchPad{Dimensions, DataMode{layer.nBytesPerIntermediateOutput}.Type, Gna2TensorModeDefault, layer.pOutputsIntermediate},
    Grouping { getGrouping(layer) },
    ElementCount { getElementCount(layer) }
{
    const auto caps = static_cast<const TensorLimits*>(validator->Capabilities);
    validator->ValidateBufferIfSet(ScratchPad.Buffer, ScratchPad.Size,  caps->Align);
    Expect::True(GNA_INT32 == ScratchPad.Mode, Gna2StatusXnnErrorIntOutputBytes);
}

//TODO:3:P1: Generalize instead addressing output at index 1
LayerOutput::LayerOutput(const Gna2Operation &operation, const LayerValidator& validatorIn) :
    Tensor{ Shape::Create(operation.Operands[1]->Shape,  capabilities.GetOrder(validatorIn)),
        operation.Operands[1]->Type, operation.Operands[1]->Data,
        Validator{ validatorIn, capabilities } },
    ScratchPad{Dimensions, Mode.Type, Mode.Mode, nullptr}, //TODO:3:P1:Decide what to do with scratch pad in API2, disabled validation, as parameters are provided by library
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}

bool LayerOutput::IsTensorValid(const Gna2Tensor &apiTensor,
      const BaseValidator& validatorIn, nn_operation operation)
{
   auto layerValidator = LayerValidator{validatorIn, operation};
   try
   {
      Tensor {
         apiTensor, capabilities.GetOrder(layerValidator),
         Validator{ layerValidator, capabilities } };
      return true;
   }
   catch (const GnaException&)
   {
      return false;
   }
}

std::pair<uint32_t, uint32_t> LayerOutput::getGroupingAndElements(
      const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
    switch (operation.Type)
    {
    case Gna2OperationTypeTransposition:
    {
        const auto& inputTensor = *operation.Operands[1];
        if (IsTensorValid(inputTensor, validatorIn, INTEL_INTERLEAVE))
        {
            return {Dimensions.at('W'), Dimensions.at('H')};
        }
        if (IsTensorValid(inputTensor, validatorIn, INTEL_DEINTERLEAVE))
        {
            return {Dimensions.at('H'), Dimensions.at('W')};
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    default:
        return Tensor::getGroupingAndElements(operation, validatorIn);
    }
}

std::pair<uint32_t, uint32_t> LayerOutput::getGroupingAndElements(const nn_layer& layer) const
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_INTERLEAVE:
     return {layer.nOutputColumns, layer.nOutputRows};
    case INTEL_GMM:
    case INTEL_COPY:
    case INTEL_RECURRENT:
    case INTEL_DEINTERLEAVE:
    case INTEL_CONVOLUTIONAL:
    case INTEL_CONVOLUTIONAL_2D:
        return {layer.nOutputRows, layer.nOutputColumns};
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}
