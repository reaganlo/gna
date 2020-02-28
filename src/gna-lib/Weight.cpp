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

#include "Weight.h"

#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"

using namespace GNA;

const FullCapabilitiesMap WeightTensor::capabilities =
{
    // TODO:3: add caps for previous device versions
    {INTEL_AFFINE, {
        AffineLayerCapabilities::GetOperands(FilterOperandIndex).at(INTEL_AFFINE)
    }},
    {INTEL_AFFINE_DIAGONAL, {
        AffineLayerCapabilities::GetOperands(FilterOperandIndex).at(INTEL_AFFINE_DIAGONAL)
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        AffineLayerCapabilities::GetOperands(FilterOperandIndex).at(INTEL_AFFINE_MULTIBIAS)
    }},
    {INTEL_CONVOLUTIONAL, {
        ConvolutionalLayer2DCapabilities::GetOperands(FilterOperandIndex).at(INTEL_CONVOLUTIONAL)
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        ConvolutionalLayer2DCapabilities::GetOperands(FilterOperandIndex).at(INTEL_CONVOLUTIONAL_2D)
    }},
    {INTEL_CONVOLUTIONAL_1D, {
        ConvolutionalLayer2DCapabilities::GetOperands(FilterOperandIndex).at(INTEL_CONVOLUTIONAL_1D)
    }},
    {INTEL_GMM, {
        GmmLayerCapabilities::GetOperands(WeightOperandIndex).at(INTEL_GMM)
    }},
    {INTEL_RECURRENT, {
        AffineLayerCapabilities::GetOperands(WeightOperandIndex).at(INTEL_RECURRENT)
    }},
};

WeightTensor::WeightTensor(const Shape& dimensions, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn)
try :
    Tensor{ dimensions, dataMode, buffer, Validator{validatorIn, capabilities} }
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, WeightOperandIndex);
}

WeightTensor::WeightTensor(const Gna2Tensor &apiTensor, const LayerValidator& validatorIn)
try :
    Tensor(apiTensor, capabilities.GetOrder(validatorIn), Validator{ validatorIn, capabilities })
{
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, WeightOperandIndex);
}
