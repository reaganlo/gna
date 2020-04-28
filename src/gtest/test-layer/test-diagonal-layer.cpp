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

#include "test-diagonal-layer.h"

#include "Layer.h"
#include "Validator.h"

#include "gna2-model-api.h"
#include "gna2-inference-api.h"
#include "gna2-inference-impl.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace GNA;

TestDiagonalLayer::TestDiagonalLayer()
{
    alignedInput = (int16_t *) _kernel_malloc(sizeof(input));
    if (alignedInput == nullptr)
    {
        this->~TestDiagonalLayer();
        throw;
    }

    alignedOutput = (int16_t *) _kernel_malloc(sizeof(refOutput));
    if (alignedOutput == nullptr)
    {
        this->~TestDiagonalLayer();
        throw;
    }

    alignedIntermediateOutput = (int32_t *) _kernel_malloc(
            sizeof(*alignedIntermediateOutput) * numberOfVectors * outputVolume);
    if (alignedIntermediateOutput == nullptr)
    {
        this->~TestDiagonalLayer();
        throw;
    }

    alignedWeight = (int16_t *) _kernel_malloc(sizeof(weight));
    if (alignedWeight == nullptr)
    {
        this->~TestDiagonalLayer();
        throw;
    }

    alignedBias = (int32_t *) _kernel_malloc(sizeof(bias));
    if (alignedBias == nullptr)
    {
        this->~TestDiagonalLayer();
        throw;
    }

    alignedPwlSegments = (intel_pwl_segment_t *)
        _kernel_malloc(sizeof(*alignedPwlSegments) * numberOfSegments);
    if (alignedPwlSegments == nullptr)
    {
        this->~TestDiagonalLayer();
        throw;
    }
}

TestDiagonalLayer::~TestDiagonalLayer()
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

const int32_t TestDiagonalLayer::bias[outputVolume]
{
        5, 4, -2, 5, -7, -5, 4, -1, 5, 4, -2, 5, -7, -5, 4, -1
};

const int16_t TestDiagonalLayer::weight[outputVolume]
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
};

const int32_t TestDiagonalLayer::refOutput[numberOfVectors * outputVolume]
{
     35, -49,  47, -19,
     -6,  12,  18,  -4,
     -2,  -9,  -3,   5,
      4,  -1,  -2,  -4,
    -11,   1, -25, -23,
    -50, -14,  13,  76,
    -44, -44,  52,  10,
    -36,   9,  -6,  -6,
    -13,  -5, -11,  15,
      4,   0,  16,  40,
     -2, -10,  -3,   0,
    -40,  45,   5, -30,
     11,   9,  -5,   1,
      7,  23,   3, -17,
      4,   4,   4,   4,
    -37, -55, -73, -19
};

const int16_t TestDiagonalLayer::input[numberOfVectors * inputVolume] =
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

TEST_F(TestDiagonalLayer, AffineDiagonalTest2B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(inputVolume, numberOfVectors, Gna2DataTypeInt16, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(inputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input, sizeof(input));

    auto outputTensor = Gna2TensorInit2D(outputVolume, numberOfVectors, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(outputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);

    auto weightTensor = Gna2TensorInit1D(outputVolume, Gna2DataTypeInt16, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(weightTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "H", 2);
    memcpy_s(alignedWeight, sizeof(int16_t) * outputVolume, weight, sizeof(weight));

    auto biasTensor = Gna2TensorInit1D(outputVolume, Gna2DataTypeInt32, alignedBias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(biasTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "H", 2);
    memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias, sizeof(bias));

    auto status = Gna2OperationInitElementWiseAffine(&affineOperation, GnaMalloc,
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

