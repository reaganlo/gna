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

#include "test-transpose-layer.h"

#include "Layer.h"
#include "TransposeLayer.h"

#include "KernelArguments.h"

#include "gna2-model-api.h"

#include "gtest/gtest.h"

using namespace GNA;

constexpr uint16_t TestTransposeLayer::numberOfElements;
const int16_t TestTransposeLayer::flatInput[numberOfElements] =
{
    -15, -14, -13, -12, -11, -10, -9,  -8,
    -7, -6, -5, -4, -3, -2, -1,  0,
     1,  2,  3,  4,  5,  6,  7,  8,
     9, 10, 11, 12, 13, 14, 15, 16
};
const int16_t TestTransposeLayer::interleaveInput[numberOfElements] =
{
    -15, -7, 1, 9,
    -14, -6, 2, 10,
    -13, -5, 3, 11,
    -12, -4, 4, 12,
    -11, -3, 5, 13,
    -10, -2, 6, 14,
    -9,  -1, 7, 15,
    -8,   0, 8, 16
};

TEST_F(TestTransposeLayer, InterleaveTest2BLegacy)
{
    intel_nnet_layer_t apiLayer;
    memset(&apiLayer, 1, sizeof(apiLayer));

    apiLayer.operation = INTEL_INTERLEAVE;
    apiLayer.mode = gna_layer_mode::INTEL_HIDDEN;
    apiLayer.pLayerStruct = nullptr;

    apiLayer.nBytesPerInput = 2;
    apiLayer.nInputRows = 4;
    apiLayer.nInputColumns = 8;

    auto inputSize = apiLayer.nBytesPerInput * apiLayer.nInputRows * apiLayer.nInputColumns;
    memcpy_s(alignedInput, inputSize, flatInput, sizeof(flatInput));
    apiLayer.pInputs = alignedInput;

    apiLayer.nBytesPerOutput = 2;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = 8;
    apiLayer.nOutputColumns = 4;
    apiLayer.pOutputsIntermediate = nullptr;

    auto outputSize = apiLayer.nBytesPerOutput * apiLayer.nOutputRows * apiLayer.nOutputColumns;
    memset(alignedOutput, 0, outputSize);
    apiLayer.pOutputs = alignedOutput;

    auto interlaveLayer = std::make_unique<TransposeLayer>(apiLayer, emptyValidator);
    ASSERT_NE(interlaveLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ nullptr, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);
    interlaveLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(alignedOutput, interleaveInput);
}

TEST_F(TestTransposeLayer, DeinterleaveTest2BLegacy)
{
    intel_nnet_layer_t apiLayer;
    memset(&apiLayer, 1, sizeof(apiLayer));

    apiLayer.operation = INTEL_DEINTERLEAVE;
    apiLayer.mode = gna_layer_mode::INTEL_HIDDEN;
    apiLayer.pLayerStruct = nullptr;

    apiLayer.nBytesPerInput = 2;
    apiLayer.nInputRows = 8;
    apiLayer.nInputColumns = 4;

    auto inputSize = apiLayer.nBytesPerInput * apiLayer.nInputRows * apiLayer.nInputColumns;
    memcpy_s(alignedInput, inputSize, interleaveInput, sizeof(flatInput));
    apiLayer.pInputs = alignedInput;

    apiLayer.nBytesPerOutput = 2;
    apiLayer.nBytesPerIntermediateOutput = 4;
    apiLayer.nOutputRows = 4;
    apiLayer.nOutputColumns = 8;
    apiLayer.pOutputsIntermediate = nullptr;

    auto outputSize = apiLayer.nBytesPerOutput * apiLayer.nOutputRows * apiLayer.nOutputColumns;
    memset(alignedOutput, 0, outputSize);
    apiLayer.pOutputs = alignedOutput;

    auto deinterlaveLayer = std::make_unique<TransposeLayer>(apiLayer, emptyValidator);
    ASSERT_NE(deinterlaveLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ nullptr, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);
    deinterlaveLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(alignedOutput, flatInput);
}

TEST_F(TestTransposeLayer, InterleaveTest2B)
{
    Gna2Operation interleaveOperation;
    memset(&interleaveOperation, 0, sizeof(interleaveOperation));

    Gna2Shape inputShape = Gna2ShapeInit2D(4, 8);
    Gna2Shape outputShape = Gna2ShapeInit2D(8, 4);

    Gna2Tensor inputTensor;
    auto inputSize = sizeof(int16_t) * numberOfElements;
    memcpy_s(alignedInput, inputSize, flatInput, sizeof(flatInput));
    inputTensor.Data = (void *) alignedInput;
    inputTensor.Mode = Gna2TensorModeDefault;
    inputTensor.Type = Gna2DataTypeInt16;
    inputTensor.Shape = inputShape;

    Gna2Tensor outputTensor;
    auto outputSize = sizeof(int16_t) * numberOfElements;
    memset(alignedOutput, 0, outputSize);
    outputTensor.Data = (void *) alignedOutput;
    outputTensor.Mode = Gna2TensorModeDefault;
    outputTensor.Type = Gna2DataTypeInt16;
    outputTensor.Shape = outputShape;

    Gna2Status status = Gna2OperationInitInterleave(
                    &interleaveOperation, GnaMalloc, &inputTensor, &outputTensor);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto interlaveLayer = std::make_unique<TransposeLayer>(interleaveOperation, emptyValidator);
    ASSERT_NE(interlaveLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ nullptr, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);
    interlaveLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(alignedOutput, interleaveInput);
}

TEST_F(TestTransposeLayer, DeinterleaveTest2B)
{
    Gna2Operation deinterleaveOperation;
    memset(&deinterleaveOperation, 0, sizeof(deinterleaveOperation));

    Gna2Shape inputShape = Gna2ShapeInit2D(8, 4);
    Gna2Shape outputShape = Gna2ShapeInit2D(4, 8);

    Gna2Tensor inputTensor;
    auto inputSize = sizeof(int16_t) * numberOfElements;
    memcpy_s(alignedInput, inputSize, interleaveInput, sizeof(flatInput));
    inputTensor.Data = (void *) alignedInput;
    inputTensor.Mode = Gna2TensorModeDefault;
    inputTensor.Type = Gna2DataTypeInt16;
    inputTensor.Shape = inputShape;

    Gna2Tensor outputTensor;
    auto outputSize = sizeof(int16_t) * numberOfElements;
    memset(alignedOutput, 0, outputSize);
    outputTensor.Data = (void *) alignedOutput;
    outputTensor.Mode = Gna2TensorModeDefault;
    outputTensor.Type = Gna2DataTypeInt16;
    outputTensor.Shape = outputShape;

    Gna2Status status = Gna2OperationInitDeInterleave(
                    &deinterleaveOperation, GnaMalloc, &inputTensor, &outputTensor);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto deinterlaveLayer = std::make_unique<TransposeLayer>(deinterleaveOperation, emptyValidator);
    ASSERT_NE(deinterlaveLayer, nullptr);

    uint32_t saturationCount;
    auto executionConfig = ExecutionConfig{ nullptr, &saturationCount, 0 };
    auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);
    deinterlaveLayer->ComputeHidden(acceleration, executionConfig);

    VerifyOutputs(alignedOutput, flatInput);
}

