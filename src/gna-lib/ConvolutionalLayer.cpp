/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include "ConvolutionalLayer.h"

#include "Validator.h"

using namespace GNA;

FiltersConfig::FiltersConfig(const nn_layer_conv * sourceLayer, const uint32_t inputElementCount) :
    BiasSimple{sourceLayer->nBytesBias, sourceLayer->pBiases},
    Count{sourceLayer->nFilters},
    CoefficientCount{sourceLayer->nFilterCoefficients},
    Data{static_cast<uint16_t*>(sourceLayer->pFilters)}
{
    Expect::InRange(sourceLayer->nFilterRows, 1, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
    Expect::True(sourceLayer->nBytesFilterCoefficient == 2, XNN_ERR_WEIGHT_BYTES);

    Expect::InRange(Count, CNN_N_FLT_COEFF_MPLY, CNN_N_FLT_MAX, CNN_ERR_FLT_COUNT);
    Expect::MultiplicityOf(Count, CNN_N_FLT_COEFF_MPLY);

    Expect::InRange(CoefficientCount, CNN_N_FLT_COEFF_MIN, CNN_N_FLT_COEFF_MAX, XNN_ERR_LYR_CFG);
    Expect::True(CoefficientCount <= inputElementCount, XNN_ERR_LYR_CFG);
    Expect::MultiplicityOf(CoefficientCount, XNN_N_IN_ELEMS_MPLY);

    Expect::ValidBuffer(Data);
}

FeatureMaps::FeatureMaps(const nn_layer_conv * sourceLayer) :
    Count{sourceLayer->nFeatureMaps},
    RowCount{sourceLayer->nFeatureMapRows},
    ColumnCount{sourceLayer->nFeatureMapColumns},
    Stride{Count * ColumnCount} // always move 1 "row"
{
    Expect::InRange(Stride, 1, CNN_N_FLT_COEFF_MAX, CNN_ERR_FLT_STRIDE);
}

ConvolutionFunction::ConvolutionFunction(const nn_layer_conv * sourceLayer, const uint32_t inputElementCount) :
    Filters{sourceLayer, inputElementCount},
    FeatureMaps{sourceLayer},
    OutputElementsCount{(inputElementCount - Filters.CoefficientCount) / FeatureMaps.Stride + 1}
{
    auto featureCount = FeatureMaps.RowCount * FeatureMaps.Stride;
    Expect::True(featureCount >= CNN_N_FLT_COEFF_MIN, XNN_ERR_LYR_CFG);
}

PoolingFunction::PoolingFunction(const nn_layer_conv * sourceLayer) :
    Type{sourceLayer->poolType},
    Size{sourceLayer->nPoolSize},
    Stride{sourceLayer->nPoolStride}
{
    Expect::InRange(Type, INTEL_NO_POOLING, NUM_POOLING_TYPES - 1, XNN_ERR_LYR_CFG);
    if (INTEL_NO_POOLING != Type)
    {
        Expect::InRange(Size, CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, XNN_ERR_LYR_CFG);
        Expect::InRange(Stride, CNN_POOL_SIZE_MIN, CNN_POOL_SIZE_MAX, CNN_ERR_POOL_STRIDE);
    }
}

CnnLayer::CnnLayer(nn_layer const * const layer) :
    Layer(layer),
    // CNN has only 2B output with Activation always enabled
    Activation(ActivationFunction::Create(&static_cast<const nn_layer_conv*>(layer->pLayerStruct)->pwl, false)),
    Convolution{static_cast<const nn_layer_conv*>(layer->pLayerStruct), Input.ElementCount},
    Pooling{static_cast<const nn_layer_conv*>(layer->pLayerStruct)},
    sourceLayer{ static_cast<const nn_layer_conv * const>(layer->pLayerStruct)}
{
    Expect::True(Input.VectorCount == 1, XNN_ERR_GROUPING);
    Expect::True(Input.VectorCount == Output.VectorCount, XNN_ERR_GROUPING);

    Output.SetOutputMode(Activation.operator bool(), layer->nBytesPerOutput);

    auto outputElementCount = Convolution.OutputElementsCount; // INTEL_NO_POOLING use convolution outputs per filter
    if (INTEL_NO_POOLING != Pooling.Type) // use pooled outputs per filter
    {
        outputElementCount = ((Convolution.OutputElementsCount - 1) / Pooling.Stride + 1);
    }
    Expect::True(Output.ElementCount == Convolution.Filters.Count * outputElementCount, XNN_ERR_LYR_CFG);
    // NOTE: intentional const override for Output.ElementCount // TODO: consider refactoring
    (uint32_t)Output.ElementCount = outputElementCount;
}

