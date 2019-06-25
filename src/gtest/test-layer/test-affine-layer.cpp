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
    alignedInput = _kernel_malloc(sizeof(input<int16_t>));
    if (alignedInput == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedOutput = _kernel_malloc(sizeof(refOutput<int32_t>));
    if (alignedOutput == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedIntermediateOutput = _kernel_malloc(
            sizeof(int32_t) * numberOfVectors * outputVolume);
    if (alignedIntermediateOutput == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedWeight = _kernel_malloc(sizeof(weight<int16_t>));
    if (alignedWeight == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedBias = _kernel_malloc(sizeof(bias<Gna2CompoundBias>));
    if (alignedBias == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedMultibias = _kernel_malloc(sizeof(multibias));
    if (alignedMultibias == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    alignedPwlSegments = (intel_pwl_segment_t *)
        _kernel_malloc(sizeof(intel_pwl_segment_t) * numberOfSegments);
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

template<typename T>
const T TestAffineLayer::weight[TestAffineLayer::outputVolume * TestAffineLayer::inputVolume] = {
    -5, -2, -1, -1, -2,  2,  5,  5,  2,  4, -1,  5, -2, -4,  0,  2,
    -3,  3, -4,  5,  5,  3, -5, -2,  5,  0, -4, -1,  1,  5,  5, -5,
     2, -3,  5,  5, -1, -2,  5,  5, -1,  4,  3,  5, -2, -1,  5,  1,
     0, -2,  1,  0,  5, -5,  5,  4, -5,  0,  3, -2,  1,  3, -5, -2,
    -5, -3,  4, -2, -3, -5,  5,  5,  5, -2, -5, -2, -5, -3, -5, -2,
    -4,  0,  5, -3, -1, -5,  4,  1, -4, -5, -3,  5,  2, -2,  2,  2,
     0, -2,  5, -3,  5, -2, -1, -3, -5,  5,  5,  5, -3,  0, -4,  2,
     2,  5, -3, -5,  3, -5, -5,  1,  5, -4, -4,  2, -5, -5,  5, -5
};

template<typename T>
const T TestAffineLayer::input[TestAffineLayer::numberOfVectors * TestAffineLayer::inputVolume] =
{
    -5,  2, -5,  4,
     5, -4, -5,  4,
     0,  5,  1, -5,
     1,  5,  5,  2,
     2, -4,  2,  3,
    -5, -1,  2,  2,
    -3, -3,  3,  1,
    -4,  2, -1, -1,
    -2, -5, -3,  5,
     0, -1,  3,  2,
     0,  3,  1, -2,
    -2,  3,  0, -5,
    -2, -3, -1, -4,
    -3, -5, -2,  3,
    -3,  0,  1,  3,
    -4, -5, -3, -2
};

template<typename T>
const T TestAffineLayer::bias[TestAffineLayer::outputVolume] = {
    5, 4, -2, 5, -5, -5, 4, -1
};

const int32_t TestAffineLayer::multibias[TestAffineLayer::outputVolume * TestAffineLayer::multibiasVectorCount] = {
    0, 0,  5, 0,
    0, 0,  4, 0,
    0, 0, -2, 0,
    0, 0,  5, 0,
    0, 0, -5, 0,
    0, 0, -5, 0,
    0, 0,  4, 0,
    0, 0, -1, 0
};

template<typename T>
const T TestAffineLayer::refOutput[TestAffineLayer::numberOfVectors * TestAffineLayer::outputVolume] =
{
   -36,   8,  53,  -35,
    37, -64,  18,  126,
   -69,  96,  61,  -38,
    25,  16,  39,  -29,
    23,  15,   9,  -39,
     5,  21,  -4, -123,
    36,  50,  38,  -76,
    67,  -9, -80,   70
};

/* Tests 1B native patch - 1B input, weight and bias */
TEST_F(TestAffineLayer, AffineTest1BLegacy)
{
    intel_nnet_layer_t apiLayer;
    memset(&apiLayer, 0, sizeof(apiLayer));

    apiLayer.mode = INTEL_HIDDEN;
    apiLayer.operation = INTEL_AFFINE;

    apiLayer.nBytesPerInput = 1;
    apiLayer.nInputRows = inputVolume;
    apiLayer.nInputColumns = numberOfVectors;
    apiLayer.pInputs = alignedInput;
    memcpy_s(alignedInput, sizeof(int8_t) * numberOfVectors * inputVolume,
            input<int8_t>, sizeof(input<int8_t>));

    apiLayer.nBytesPerOutput = 4;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = outputVolume;
    apiLayer.nOutputColumns = numberOfVectors;
    apiLayer.pOutputsIntermediate = alignedIntermediateOutput;
    apiLayer.pOutputs = alignedOutput;

    intel_affine_func_t affineFunc;
    affineFunc.nBytesPerWeight = 1;
    affineFunc.pWeights = alignedWeight;
    memcpy_s(alignedWeight, sizeof(int8_t) * inputVolume * outputVolume, weight<int8_t>, sizeof(weight<int8_t>));
    affineFunc.nBytesPerBias = 1;
    affineFunc.pBiases = alignedBias;
    memcpy_s(alignedBias, sizeof(int8_t) * outputVolume, bias<int8_t>, sizeof(bias<int8_t>));

    intel_pwl_func_t pwl;
    pwl.nSegments = 0;
    pwl.pSegments = nullptr;

    intel_affine_layer_t layer;
    layer.affine = affineFunc;
    layer.pwl = pwl;

    apiLayer.pLayerStruct = &layer;

    auto affineLayer = Layer::Create(apiLayer, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount = 0;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    EXPECT_EQ(saturationCount, 0);

    VerifyOutputs(static_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
                             numberOfVectors * outputVolume);
}

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
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume,
            input<int16_t>, sizeof(input<int16_t>));

    apiLayer.nBytesPerOutput = 4;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = outputVolume;
    apiLayer.nOutputColumns = numberOfVectors;
    apiLayer.pOutputsIntermediate = alignedIntermediateOutput;
    apiLayer.pOutputs = alignedOutput;

    intel_affine_func_t affineFunc;
    affineFunc.nBytesPerWeight = 2;
    affineFunc.pWeights = alignedWeight;
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight<int16_t>, sizeof(weight<int16_t>));
    affineFunc.nBytesPerBias = 4;
    affineFunc.pBiases = alignedBias;
    memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias<int32_t>, sizeof(bias<int32_t>));

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

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
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
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume,
             input<int16_t>, sizeof(input<int16_t>));

    apiLayer.nBytesPerOutput = 4;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = outputVolume;
    apiLayer.nOutputColumns = numberOfVectors;
    apiLayer.pOutputsIntermediate = alignedIntermediateOutput;
    apiLayer.pOutputs = alignedOutput;

    intel_affine_multibias_func_t affineFunc;
    affineFunc.nBytesPerWeight = 2;
    affineFunc.pWeights = alignedWeight;
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight<int16_t>, sizeof(weight<int16_t>));
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

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
                  numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineTest1B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(inputVolume, numberOfVectors, Gna2DataTypeInt8, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy(inputTensor.Layout, "HW", 3);
    memcpy_s(alignedInput, sizeof(int8_t) * numberOfVectors * inputVolume,
             input<int8_t>, sizeof(input<int8_t>));

    auto outputTensor = Gna2TensorInit2D(outputVolume, numberOfVectors, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy(outputTensor.Layout, "HW", 3);

    auto weightTensor = Gna2TensorInit2D(outputVolume, inputVolume, Gna2DataTypeInt8, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy(weightTensor.Layout, "HW", 3);
    memcpy_s(alignedWeight, sizeof(int8_t) * inputVolume * outputVolume, weight<int8_t>, sizeof(weight<int8_t>));

    auto biasTensor = Gna2TensorInit1D(outputVolume, Gna2DataTypeInt8, alignedBias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy(biasTensor.Layout, "H", 2);
    memcpy_s(alignedBias, sizeof(int8_t) * outputVolume, bias<int8_t>, sizeof(bias<int8_t>));

    auto status = Gna2OperationInitFullyConnectedAffine(&affineOperation, GnaMalloc,
            &inputTensor, &outputTensor, &weightTensor, &biasTensor, nullptr);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto affineLayer = Layer::Create(affineOperation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
                  numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineTest2B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(inputVolume, numberOfVectors, Gna2DataTypeInt16, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(inputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume,
             input<int16_t>, sizeof(input<int16_t>));

    auto outputTensor = Gna2TensorInit2D(outputVolume, numberOfVectors, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(outputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);

    auto weightTensor = Gna2TensorInit2D(outputVolume, inputVolume, Gna2DataTypeInt16, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(weightTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight<int16_t>, sizeof(weight<int16_t>));

    auto biasTensor = Gna2TensorInit1D(outputVolume, Gna2DataTypeInt32, alignedBias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(biasTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "H", 2);
    memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias<int32_t>, sizeof(bias<int32_t>));

    auto status = Gna2OperationInitFullyConnectedAffine(&affineOperation, GnaMalloc,
            &inputTensor, &outputTensor, &weightTensor, &biasTensor, nullptr);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto affineLayer = Layer::Create(affineOperation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
                  numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineMultibiasTest2B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(inputVolume, numberOfVectors, Gna2DataTypeInt16, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(inputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume,
             input<int16_t>, sizeof(input<int16_t>));

    auto outputTensor = Gna2TensorInit2D(outputVolume, numberOfVectors, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(outputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);

    auto weightTensor = Gna2TensorInit2D(outputVolume, inputVolume, Gna2DataTypeInt16, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(weightTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    strncpy(weightTensor.Layout, "HW", 3);
    memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight<int16_t>, sizeof(weight<int16_t>));

    auto biasTensor = Gna2TensorInit2D(outputVolume, multibiasVectorCount, Gna2DataTypeInt32, alignedMultibias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(biasTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
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

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
                  numberOfVectors * outputVolume);
}

