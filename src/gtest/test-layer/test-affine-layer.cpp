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

#include "test-affine-layer.h"
#include "test-layer.h"

#include "Layer.h"
#include "AffineLayers.h"

#include "KernelArguments.h"

#include "gna2-model-api.h"

#include "gtest/gtest.h"

using namespace GNA;

TestAffineLayer::TestAffineLayer()
{
    alignedInput = (int16_t *) _kernel_malloc(sizeof(input));
    if (alignedInput == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedOutput = (int16_t *) _kernel_malloc(sizeof(refOutput));
    if (alignedOutput == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedIntermediateOutput = (int32_t *) _kernel_malloc(
            sizeof(*alignedIntermediateOutput) * numberOfVectors * outputVolume);
    if (alignedIntermediateOutput == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedWeight = (int16_t *) _kernel_malloc(sizeof(weight));
    if (alignedWeight == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedBias = (int32_t *) _kernel_malloc(sizeof(bias));
    if (alignedBias == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedMultibias = (int32_t *) _kernel_malloc(sizeof(multibias));
    if (alignedMultibias == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedPwlSegments = (intel_pwl_segment_t *)
        _kernel_malloc(sizeof(*alignedPwlSegments) * numberOfSegments);
    if (alignedPwlSegments == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }
}

TestAffineLayer::~TestAffineLayer()
{
    if (alignedInput != nullptr)
    {
        _gna_free(alignedInput);
        alignedInput = nullptr;
    }

    if (alignedOutput != nullptr)
    {
        _gna_free(alignedOutput);
        alignedOutput = nullptr;
    }

    if (alignedWeight != nullptr)
    {
        _gna_free(alignedWeight);
        alignedWeight = nullptr;
    }

    if (alignedBias != nullptr)
    {
        _gna_free(alignedBias);
        alignedBias = nullptr;
    }

    if (alignedMultibias != nullptr)
    {
        _gna_free(alignedMultibias);
        alignedMultibias = nullptr;
    }

    if (alignedIntermediateOutput != nullptr)
    {
        _gna_free(alignedIntermediateOutput);
        alignedIntermediateOutput = nullptr;
    }

    if (alignedPwlSegments != nullptr)
    {
        _gna_free(alignedPwlSegments);
        alignedPwlSegments = nullptr;
    }
}

const int16_t TestAffineLayer::weight[outputVolume * inputVolume] = {
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t TestAffineLayer::input[numberOfVectors * inputVolume] =
{
    -5,  9, -7,  4,
     5, -4, -7,  4,
     0,  7,  1, -7,
     1,  6,  7,  9,
     2, -4,  9,  8,
    -5, -1,  2,  9,
    -8, -8,  8,  1,
    -7,  2, -1, -1,
    -9, -5, -8,  5,
     0, -1,  3,  9,
     0,  8,  1, -2,
    -9,  8,  0, -7,
    -9, -8, -1, -4,
    -3, -7, -2,  3,
    -8,  0,  1,  3,
    -4, -6, -8, -2
};

const int32_t TestAffineLayer::bias[outputVolume] = {
    5, 4, -2, 5, -7, -5, 4, -1
};

const int32_t TestAffineLayer::multibias[outputVolume * multibiasVectorCount] = {
    0, 0,  5, 0,
    0, 0,  4, 0,
    0, 0, -2, 0,
    0, 0,  5, 0,
    0, 0, -7, 0,
    0, 0, -5, 0,
    0, 0,  4, 0,
    0, 0, -1, 0
};

const int32_t TestAffineLayer::refOutput[numberOfVectors * outputVolume] =
{
    -177,  -85,   29,   28,
      96, -173,   25,  252,
    -160,  274,  157,  -29,
      48,  -60,  158,  -29,
      26,   -2,  -44, -251,
    -173,  -70,   -1, -323,
      99,  144,   38,  -63,
      20,   56, -103,   10
};

TEST_F(TestAffineLayer, AffineTest2BLegacy)
{
    intel_nnet_layer_t apiLayer;
    memset(&apiLayer, 0, sizeof(apiLayer));

    apiLayer.mode = INTEL_HIDDEN;
    apiLayer.operation = INTEL_AFFINE;

    apiLayer.nBytesPerInput = 2;
    apiLayer.nInputRows = inputVolume;
    apiLayer.nInputColumns = numberOfVectors;
    apiLayer.pInputs = alignedInput;
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input, sizeof(input));

    apiLayer.nBytesPerOutput = 4;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = outputVolume;
    apiLayer.nOutputColumns = numberOfVectors;
    apiLayer.pOutputsIntermediate = alignedIntermediateOutput;
    apiLayer.pOutputs = alignedOutput;

    intel_affine_func_t affineFunc;
    affineFunc.nBytesPerWeight = 2;
    affineFunc.pWeights = alignedWeight;
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight, sizeof(weight));
    affineFunc.nBytesPerBias = 4;
    affineFunc.pBiases = alignedBias;
    memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias, sizeof(bias));

    intel_pwl_func_t pwl;
    pwl.nSegments = 0;
    pwl.pSegments = nullptr;

    intel_affine_layer_t layer;
    layer.affine = affineFunc;
    layer.pwl = pwl;

    apiLayer.pLayerStruct = &layer;

    auto affineLayer = Layer::Create(apiLayer, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput,
                             numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineMultibiasTest2BLegacy)
{
    intel_nnet_layer_t apiLayer;
    memset(&apiLayer, 0, sizeof(apiLayer));

    apiLayer.mode = INTEL_HIDDEN;
    apiLayer.operation = INTEL_AFFINE_MULTIBIAS;

    apiLayer.nBytesPerInput = 2;
    apiLayer.nInputRows = inputVolume;
    apiLayer.nInputColumns = numberOfVectors;
    apiLayer.pInputs = alignedInput;
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input, sizeof(input));

    apiLayer.nBytesPerOutput = 4;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = outputVolume;
    apiLayer.nOutputColumns = numberOfVectors;
    apiLayer.pOutputsIntermediate = alignedIntermediateOutput;
    apiLayer.pOutputs = alignedOutput;

    intel_affine_multibias_func_t affineFunc;
    affineFunc.nBytesPerWeight = 2;
    affineFunc.pWeights = alignedWeight;
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight, sizeof(weight));
    affineFunc.nBytesPerBias = 4;
    affineFunc.pBiases = alignedMultibias;
    auto multibiasSize = sizeof(int32_t) * outputVolume * multibiasVectorCount;
    memcpy_s(alignedMultibias, multibiasSize, multibias, sizeof(multibias));
    affineFunc.biasVectorCount = multibiasVectorCount;
    affineFunc.biasVectorIndex = 2;
    affineFunc.weightScaleFactors = nullptr;

    intel_pwl_func_t pwl;
    pwl.nSegments = 0;
    pwl.pSegments = nullptr;

    intel_affine_multibias_layer_t layer;
    layer.affine = affineFunc;
    layer.pwl = pwl;

    apiLayer.pLayerStruct = &layer;

    auto affineLayer = Layer::Create(apiLayer, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput,
                  numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineTest2B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(inputVolume, numberOfVectors, Gna2DataTypeInt16, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy(inputTensor.Layout, "HW", 3);
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input, sizeof(input));

    auto outputTensor = Gna2TensorInit2D(outputVolume, numberOfVectors, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy(outputTensor.Layout, "HW", 3);

    auto weightTensor = Gna2TensorInit2D(outputVolume, inputVolume, Gna2DataTypeInt16, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy(weightTensor.Layout, "HW", 3);
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight, sizeof(weight));

    auto biasTensor = Gna2TensorInit1D(outputVolume, Gna2DataTypeInt32, alignedBias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy(biasTensor.Layout, "H", 2);
    memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias, sizeof(bias));

    auto status = Gna2OperationInitFullyConnectedAffine(&affineOperation, GnaMalloc,
            &inputTensor, &outputTensor, &weightTensor, &biasTensor, nullptr);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto affineLayer = Layer::Create(affineOperation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput,
                  numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineMultibiasTest2B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(inputVolume, numberOfVectors, Gna2DataTypeInt16, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy(inputTensor.Layout, "HW", 3);
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input, sizeof(input));

    auto outputTensor = Gna2TensorInit2D(outputVolume, numberOfVectors, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy(outputTensor.Layout, "HW", 3);

    auto weightTensor = Gna2TensorInit2D(outputVolume, inputVolume, Gna2DataTypeInt16, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy(weightTensor.Layout, "HW", 3);
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight, sizeof(weight));

    auto biasTensor = Gna2TensorInit2D(outputVolume, multibiasVectorCount, Gna2DataTypeInt32, alignedMultibias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy(biasTensor.Layout, "HW", 3);
    auto multibiasSize = sizeof(int32_t) * outputVolume * multibiasVectorCount;
    memcpy_s(alignedMultibias, multibiasSize, multibias, sizeof(multibias));

    Gna2BiasMode biasMode;
    uint32_t biasVectorIndex = 2;

    auto status = Gna2OperationInitFullyConnectedBiasGrouping(&affineOperation, GnaMalloc,
            &inputTensor, &outputTensor, &weightTensor, &biasTensor, nullptr,
            nullptr, &biasMode, &biasVectorIndex);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto affineLayer = Layer::Create(affineOperation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput,
                  numberOfVectors * outputVolume);
}

