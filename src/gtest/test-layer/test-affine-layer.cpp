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
    auto const i = &input<int8_t>;
    auto const i2 = &input<int16_t>;
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
    auto const w = &weight<int8_t>;
    auto const w2 = &weight<int16_t>;
    alignedWeight = _kernel_malloc(sizeof(weight<int16_t>));
    if (alignedWeight == nullptr)
    {
        this->~TestAffineLayer();
        throw;
    }

    auto const b = &bias<int8_t>;
    auto const b2 = &bias<int16_t>;
    auto const b3 = &bias<int32_t>;
    alignedBias = _kernel_malloc(sizeof(bias<int32_t>));
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

    if (operation.Operands != nullptr)
    {
        free(operation.Operands);
    }

    if (operation.Parameters != nullptr)
    {
        free(operation.Parameters);
    }
    memset(&operation, 0, sizeof(operation));
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

void TestAffineLayer::ExecuteAffineTest() const
{
    auto const affineLayer = Layer::Create(operation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    uint32_t saturationCount;
    auto const executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    auto const acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    affineLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput<int32_t>,
        numberOfVectors * outputVolume);
}

TEST_F(TestAffineLayer, AffineTest1B)
{
    RunAffineTest<int8_t, int8_t, int8_t>(Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8);
}

TEST_F(TestAffineLayer, AffineTest2B)
{
    RunAffineTest<int16_t, int16_t, int32_t>(Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt32);
}

TEST_F(TestAffineLayer, AffineTest1BInput2BWeight)
{
    RunAffineTest<int8_t, int16_t, int32_t>(Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32);
}

TEST_F(TestAffineLayer, AffineMultibiasTest2B)
{
    inputTensor = PrepareInput<int16_t>(Gna2DataTypeInt16);

    outputTensor = PrepareOutput(Gna2DataTypeInt32);

    weightTensor = PrepareWeight<int16_t>(Gna2DataTypeInt16);

    biasTensor = PrepareTensor(Gna2DataTypeInt32, alignedMultibias, "HW", outputVolume, multibiasVectorCount);
    const auto multibiasSize = sizeof(int32_t) * outputVolume * multibiasVectorCount;
    memcpy_s(alignedMultibias, multibiasSize, multibias, sizeof(multibias));

    Gna2BiasMode biasMode;
    uint32_t biasVectorIndex = 2;

    auto const status = Gna2OperationInitFullyConnectedBiasGrouping(&operation, GnaMalloc,
        &inputTensor, &outputTensor, &weightTensor, &biasTensor, nullptr,
        nullptr, &biasMode, &biasVectorIndex);
    ASSERT_EQ(status, Gna2StatusSuccess);

    ExecuteAffineTest();
}

TEST_F(TestAffineLayer, AffineTestAdlWorkaround)
{
    PrepareAffineTest<int8_t, int16_t, int32_t>(Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32);
    auto const affineLayer = Layer::Create(operation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    affineLayer->VerifyHas1BInputAnd2BWeight();

    auto const is1B2B = affineLayer->Is1BInputAnd2BWeight();
    ASSERT_EQ(true, is1B2B);
}
