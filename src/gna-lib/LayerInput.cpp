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

#include "Capabilities.h"
#include "DataMode.h"
#include "Macros.h"
#include "ParameterLimits.h"
#include "Validator.h"

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <algorithm>
#include <cstdint>
#include <memory.h>
#include <vector>

using namespace GNA;

static const std::vector<uint32_t> _Multipliers =
{ 2 * XNN_N_IN_ELEMS_MPLY, 1 * XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MPLY / 2};

static const ShapeLimits _InterleaveLimits =
{
    {GNA_DIM_H, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, Gna2StatusXnnErrorInputVolume}},
    {GNA_DIM_W, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorInputVolume}}
};

static const ShapeLimits _FlatLimits =
{
    {GNA_DIM_H, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorInputVolume}},
    {GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, Gna2StatusXnnErrorInputVolume}}
};

static const DataModeLimits _ModesGen0_9 = {
    { GNA_INT16 },
    Gna2StatusXnnErrorInputBytes
};

static const TensorLimits _InterleaveTensorLimitsGen0_9 =
{
    {GNA_TENSOR_HW},
    _InterleaveLimits,
    _ModesGen0_9
};

static const TensorLimits _FlatTensorLimitsGen0_9 =
{
    {GNA_TENSOR_HW},
    _FlatLimits,
    _ModesGen0_9
};

/* GNA_DATA_DISABLED may be supported in next generation */
static const DataModeLimits _ModesGen3 = {
    { GNA_INT8, GNA_INT16 },
    Gna2StatusXnnErrorInputBytes
};

static const TensorLimits _InterleaveTensorLimitsGen3 =
{
    {GNA_TENSOR_HW},
    _InterleaveLimits,
    _ModesGen3
};

static const TensorLimits _FlatTensorLimitsGen3 =
{
    {GNA_TENSOR_HW},
    _FlatLimits,
    _ModesGen3
};

const FullCapabilitiesMap LayerInput::capabilities =
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
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_WN },
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, Gna2StatusXnnErrorInputVolume}}},
            _ModesGen0_9})},
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        {GNA_1_0, std::make_shared<TensorLimits>(TensorLimits{
            { GNA_TENSOR_WN },
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_W, {XNN_N_IN_ELEMS_MPLY, XNN_N_IN_ELEMS_MAX, _Multipliers, Gna2StatusXnnErrorInputVolume}}},
            _ModesGen0_9})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},    // N = 1
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorInputVolume}}},
            _ModesGen3})}
    }},
   {GNA_LAYER_CNN_2D_POOLING, {
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_NHWD},    // N = #kernels + GNA_BIAS_PER_KERNEL (HWD=1) or GNA_BIAS_PER_STRIDE (HWD each filter dimensions),
            {{GNA_DIM_N, {1, 1, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_W, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_D, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorInputVolume}}},
             { { GNA_INT8, GNA_INT16 }, Gna2StatusXnnErrorInputBytes }})}
    }},
    {INTEL_COPY, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
    {INTEL_INTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }},
    {INTEL_DEINTERLEAVE, {
        {GNA_0_9, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_InterleaveTensorLimitsGen3)}
    }},
     {INTEL_GMM, {
        {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_WN},                   // H - GMM states, D - #mixtures
            {{GNA_DIM_N, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorInputVolume}},
             {GNA_DIM_W, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
            { { GNA_INT8}, Gna2StatusXnnErrorInputBytes }})}
    }},
    {INTEL_RECURRENT, {
        {GNA_0_9, std::make_shared<TensorLimits>(_FlatTensorLimitsGen0_9)},
        {GNA_3_0, std::make_shared<TensorLimits>(_FlatTensorLimitsGen3)}
    }}
};

LayerInput::LayerInput(const nn_layer &layer, const LayerValidator& validatorIn) :
    Tensor{GetDimensions(layer,  capabilities.GetOrder(validatorIn)),
        layer.nBytesPerInput, layer.pInputs,
        Validator{ validatorIn, capabilities } },
     Grouping{ getGrouping(layer) },
     ElementCount{ getElementCount(layer) }
{
}

LayerInput::LayerInput(const Gna2Operation &operation, const LayerValidator& validatorIn) :
    Tensor{ *operation.Operands[0], capabilities.GetOrder(validatorIn),
        Validator{ validatorIn, capabilities } },
    Grouping{ getGrouping(operation, validatorIn) },
    ElementCount{ getElementCount(operation, validatorIn) }
{
}

bool LayerInput::IsTensorValid(const Gna2Tensor &apiTensor,
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

Shape LayerInput::GetDimensions(const nn_layer& layer, gna_tensor_order order)
{
    switch (layer.operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_COPY:
    case INTEL_DEINTERLEAVE:
    case INTEL_INTERLEAVE:
    case INTEL_RECURRENT:
        return {order, layer.nInputRows, layer.nInputColumns};
    case INTEL_CONVOLUTIONAL:
    case INTEL_GMM:
        return {order, layer.nInputColumns, layer.nInputRows};
    case INTEL_CONVOLUTIONAL_2D:
    {
        auto const config = static_cast<nn_layer_cnn2d*>(layer.pLayerStruct);
        return {order,
            layer.nInputRows,
            config->inputDimensions.height,
            config->inputDimensions.width,
            config->inputDimensions.depth}; // GNA_TENSOR_NHWD
    }
    default:
        return {};
    }
}

uint32_t LayerInput::getGrouping(
      const Gna2Operation& operation, const LayerValidator& validatorIn) const
{
   switch (operation.Type)
   {
      case Gna2OperationTypeFullyConnectedAffine:
      case Gna2OperationTypeElementWiseAffine:
         return Dimensions.at('W');
      case Gna2OperationTypeRecurrent:
      case Gna2OperationTypeCopy:
      case Gna2OperationTypeGmm:
         return Dimensions.at('H');
      case Gna2OperationTypeConvolution:
         return Dimensions.at('N'); // not applicable for 2D CNN
      case Gna2OperationTypeTransposition:
         {
            const auto& inputTensor = *operation.Operands[0];
            if (LayerInput::IsTensorValid(inputTensor, validatorIn, INTEL_INTERLEAVE))
            {
               return Dimensions.at('H');
            }
            if (LayerInput::IsTensorValid(inputTensor, validatorIn, INTEL_DEINTERLEAVE))
            {
               return Dimensions.at('W');
            }
         }
      default:
         throw GnaException(Gna2StatusNotImplemented);
   }
}

uint32_t LayerInput::getElementCount(const Gna2Operation& operation,
      const LayerValidator& validatorIn) const
{
   switch (operation.Type)
   {
      case Gna2OperationTypeFullyConnectedAffine:
      case Gna2OperationTypeElementWiseAffine:
         return Dimensions.at('H');
      case Gna2OperationTypeRecurrent:
      case Gna2OperationTypeCopy:
      case Gna2OperationTypeGmm:
         return Dimensions.at('W');
      case Gna2OperationTypeConvolution:
         return Dimensions.at('H'); // not applicable 2D CNN
      case Gna2OperationTypeTransposition:
         {
            const auto& inputTensor = *operation.Operands[0];
            if (LayerInput::IsTensorValid(inputTensor, validatorIn, INTEL_INTERLEAVE))
            {
               return Dimensions.at('W');
            }
            if (LayerInput::IsTensorValid(inputTensor, validatorIn, INTEL_DEINTERLEAVE))
            {
               return Dimensions.at('H');
            }
         }
      default:
         throw GnaException(Gna2StatusNotImplemented);
   }
}

uint32_t LayerInput::getGrouping(const nn_layer& layer) const
{
   switch (layer.operation)
   {
      case INTEL_AFFINE:
      case INTEL_AFFINE_DIAGONAL:
      case INTEL_AFFINE_MULTIBIAS:
      case INTEL_DEINTERLEAVE:
         return layer.nInputColumns;
      case INTEL_GMM:
      case INTEL_COPY:
      case INTEL_RECURRENT:
      case INTEL_INTERLEAVE:
      case INTEL_CONVOLUTIONAL:
      case INTEL_CONVOLUTIONAL_2D:
         return layer.nInputRows;
      default:
         throw GnaException(Gna2StatusNotImplemented);
   }
}

uint32_t LayerInput::getElementCount(const nn_layer& layer) const
{
   switch (layer.operation)
   {
      case INTEL_AFFINE:
      case INTEL_AFFINE_DIAGONAL:
      case INTEL_AFFINE_MULTIBIAS:
      case INTEL_DEINTERLEAVE:
         return layer.nInputRows;
      case INTEL_GMM:
      case INTEL_COPY:
      case INTEL_RECURRENT:
      case INTEL_INTERLEAVE:
      case INTEL_CONVOLUTIONAL:
      case INTEL_CONVOLUTIONAL_2D:
         return layer.nInputColumns;
      default:
         throw GnaException(Gna2StatusNotImplemented);
   }
}

