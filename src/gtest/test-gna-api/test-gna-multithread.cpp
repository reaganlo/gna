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

#include "test-gna-multithread.h"
#include "test-gna-api.h"

#include "gna2-api.h"
#include "gna2-device-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-api.h"

#include <limits>

static void * GnaAllocator(uint32_t size)
{
    return malloc(size);
}
static void GnaDeAllocator(void * ptr)
{
    return free(ptr);
}

const Gna2PwlSegment identityPwl[] = {
    {(std::numeric_limits<int16_t>::min)(), (std::numeric_limits<int16_t>::min)(), 0x100},
    {0, 0, 0x100} };

constexpr uint32_t inputDiagonalSize = 0xf000;
constexpr uint32_t nModels = 4;
constexpr uint32_t modelBytes = 16 * inputDiagonalSize; // 16 is enough as > 2 input + 2 output + 2 weight + 4 biases + pwl
constexpr uint32_t timeoutMillis = 10000;
constexpr uint32_t totalIteration = 1000;

Gna2Tensor* allocAndSet(const Gna2Tensor& value)
{
    return &(*static_cast<Gna2Tensor *>(GnaAllocator(sizeof(Gna2Tensor))) = value);
}
Gna2Operation simpleDiagonalOperation(void* input, void* output, void* weights, void* biases, void* pwl, uint32_t diagSize, uint32_t pwlSize)
{
    Gna2Operation operation{};
    Gna2Tensor *inputTensor = allocAndSet(Gna2TensorInit2D(diagSize, 1, Gna2DataTypeInt16, input));
    Gna2Tensor *outputTensor = allocAndSet(Gna2TensorInit2D(diagSize, 1, Gna2DataTypeInt16, output));
    Gna2Tensor *weightTensor = allocAndSet(Gna2TensorInit1D(diagSize, Gna2DataTypeInt16, weights));
    Gna2Tensor *biasTensor = allocAndSet(Gna2TensorInit1D(diagSize, Gna2DataTypeInt32, biases));
    Gna2Tensor *pwlTensor = allocAndSet(Gna2TensorInit1D(pwlSize, Gna2DataTypePwlSegment, pwl));
    Gna2OperationInitElementWiseAffine(&operation, GnaAllocator, inputTensor, outputTensor, weightTensor, biasTensor, pwlTensor);
    return operation;
}
constexpr int modelValue(int modelIndex)
{
    return modelIndex + 1;
}
Gna2Model simpleModel(void* gnaMemory, int modelIndex)
{
    const auto input = static_cast<int16_t *>(gnaMemory);
    std::fill_n(input, inputDiagonalSize, modelValue(modelIndex));
    const auto output = input + inputDiagonalSize;
    std::fill_n(output, inputDiagonalSize, 0);
    const auto weights = output + inputDiagonalSize;
    std::fill_n(weights, inputDiagonalSize, modelValue(modelIndex));
    const auto biases = reinterpret_cast<int32_t *>(weights + inputDiagonalSize);
    std::fill_n(biases, inputDiagonalSize, modelValue(modelIndex));
    const auto pwl = reinterpret_cast<Gna2PwlSegment *>(biases + inputDiagonalSize);
    const auto pwlSize = sizeof(identityPwl) / sizeof(Gna2PwlSegment);
    std::copy_n(identityPwl, pwlSize, pwl);
    const Gna2Model model{ 1, static_cast<Gna2Operation*>(GnaAllocator(sizeof(Gna2Operation))) };
    *model.Operations = simpleDiagonalOperation(input, output, weights, biases, pwl, inputDiagonalSize, pwlSize);
    return model;
}

uint32_t CountErrorsInScores(void* begin, int modelIndex)
{
    const auto *input = static_cast<int16_t *>(begin);
    const auto output = input + inputDiagonalSize;
    uint32_t count = 0;
    for (uint32_t i = 0; i < inputDiagonalSize; i++)
    {
        if (output[i] != modelValue(modelIndex)*modelValue(modelIndex) + modelValue(modelIndex))
        {
            count++;
        }
    }
    return count;
}

void RunWithThreads(const uint32_t nThreads)
{
    ASSERT_EQ(Gna2StatusSuccess, Gna2DeviceOpen(0));
    ASSERT_EQ(Gna2StatusSuccess, Gna2DeviceSetNumberOfThreads(0, nThreads));
    void * memory;

    uint32_t sg;

    ASSERT_EQ(Gna2StatusSuccess, Gna2MemoryAlloc(nModels * modelBytes, &sg, &memory));
    Gna2Model model[nModels];
    uint32_t modelId[nModels];
    uint32_t reqCfgId[nModels];
    uint32_t reqId[nModels];
    for (uint32_t modelIndex = 0; modelIndex < nModels; modelIndex++)
    {
        const auto thisModel = static_cast<uint8_t*>(memory) + modelBytes * modelIndex;
        model[modelIndex] = simpleModel(thisModel, modelIndex);
        ASSERT_EQ(Gna2ModelCreate(0, model + modelIndex, modelId + modelIndex), Gna2StatusSuccess);
        ASSERT_EQ(Gna2RequestConfigCreate(modelId[modelIndex], reqCfgId + modelIndex), Gna2StatusSuccess);
    }
    for (uint32_t i = 0; i < totalIteration; i++)
    {
        for (uint32_t modelIndex = 0; modelIndex < nModels; modelIndex++)
        {
            ASSERT_EQ(Gna2RequestEnqueue(reqCfgId[modelIndex], reqId + modelIndex), Gna2StatusSuccess);
        }

        for (uint32_t modelIndex = 0; modelIndex < nModels; modelIndex++)
        {
            ASSERT_EQ(Gna2RequestWait(reqId[modelIndex], timeoutMillis), Gna2StatusSuccess);
            const auto thisModel = static_cast<uint8_t*>(memory) + modelBytes * modelIndex;
            ASSERT_EQ(0, CountErrorsInScores(thisModel, modelIndex)) << "Error at, iteration: " << i << " model number: " << modelIndex;
        }
    }
    ASSERT_EQ(Gna2StatusSuccess, Gna2MemoryFree(memory));
    ASSERT_EQ(Gna2StatusSuccess, Gna2DeviceClose(0));
}

TEST_F(TestGnaApi, multipleDiagModelCreateRunWithThreads1)
{
    RunWithThreads(1);
}

TEST_F(TestGnaApi, DISABLED_multipleDiagModelCreateRunWithThreads2)
{
    RunWithThreads(2);
}