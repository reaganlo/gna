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

#include "test-recurrent-layer.h"

#include "Layer.h"
#include "AffineLayers.h"
#include "RecurrentLayer.h"

#include "KernelArguments.h"

#include "gna2-model-api.h"

#include "gtest/gtest.h"

using namespace GNA;

TestRecurrentLayer::TestRecurrentLayer()
{
    alignedInput = static_cast<int16_t *>(_kernel_malloc(sizeof(input)));
    if (alignedInput == nullptr)
    {
        this->~TestRecurrentLayer();
        throw;
    }

    alignedOutput = static_cast<int16_t *>(_kernel_malloc(sizeof(refOutput)));
    if (alignedOutput == nullptr)
    {
        this->~TestRecurrentLayer();
        throw;
    }

    alignedIntermediateOutput = static_cast<int32_t *>(_kernel_malloc(
            sizeof(*alignedIntermediateOutput) * numberOfVectors * outputVolume));
    if (alignedIntermediateOutput == nullptr)
    {
        this->~TestRecurrentLayer();
        throw;
    }

    alignedWeight = static_cast<int16_t *>(_kernel_malloc(sizeof(weight)));
    if (alignedWeight == nullptr)
    {
        this->~TestRecurrentLayer();
        throw;
    }

    alignedBias = static_cast<int32_t *>(_kernel_malloc(sizeof(bias)));
    if (alignedBias == nullptr)
    {
        this->~TestRecurrentLayer();
        throw;
    }

    alignedPwlSegments = _kernel_malloc(sizeof(pwlSegments));
    if (alignedPwlSegments == nullptr)
    {
        this->~TestRecurrentLayer();
        throw;
    }
}

TestRecurrentLayer::~TestRecurrentLayer()
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

const int16_t TestRecurrentLayer::input[numberOfVectors * inputVolume]
{
};

const int16_t TestRecurrentLayer::weight[(inputVolume + outputVolume) * outputVolume]
{
};

const int32_t TestRecurrentLayer::bias[outputVolume]
{
};

const int16_t TestRecurrentLayer::refOutput[numberOfVectors * outputVolume]
{
};

TEST_F(TestRecurrentLayer, RecurrentTest2B)
{
    Gna2Operation affineOperation;
    memset(&affineOperation, 0, sizeof(affineOperation));

    auto inputTensor = Gna2TensorInit2D(numberOfVectors, inputVolume, Gna2DataTypeInt16, alignedInput);
    inputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(inputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input, sizeof(input));

    auto outputTensor = Gna2TensorInit2D(numberOfVectors, outputVolume, Gna2DataTypeInt32, alignedOutput);
    outputTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(outputTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);

    auto weightTensor = Gna2TensorInit2D(outputVolume, inputVolume + outputVolume, Gna2DataTypeInt16, alignedWeight);
    weightTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(weightTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "HW", 3);
    memcpy_s(alignedWeight, sizeof(int16_t) * (inputVolume + outputVolume) * outputVolume, weight, sizeof(weight));

    auto biasTensor = Gna2TensorInit1D(outputVolume, Gna2DataTypeInt32, alignedBias);
    biasTensor.Mode = Gna2TensorModeDefault;
    strncpy_s(biasTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "H", 2);
    memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias, sizeof(bias));

    samplePwl(static_cast<PwlSegment *>(alignedPwlSegments), numberOfSegments);
    auto pwlTensor = Gna2TensorInitActivation(numberOfSegments,
                        static_cast<Gna2PwlSegment *>(alignedPwlSegments));
    strncpy_s(pwlTensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, "H", 2);

    uint32_t delay = 2;
    auto status = Gna2OperationInitRecurrent(&affineOperation, GnaMalloc, &inputTensor,
            &outputTensor, &weightTensor, &biasTensor, &pwlTensor, &delay);
    ASSERT_EQ(status, Gna2StatusSuccess);

    auto affineLayer = Layer::Create(affineOperation, emptyValidator);
    ASSERT_NE(affineLayer, nullptr);

    // TODO: uncomment when internally allocated scratchpad will be implemented
    //uint32_t saturationCount;
    //auto executionConfig = ExecutionConfig{ &kernelBuffers, &saturationCount, 0 };
    //auto acceleration = AccelerationMode(Gna2AccelerationModeGeneric, false);

    //affineLayer->ComputeHidden(acceleration, executionConfig);

    //VerifyOutputs(reinterpret_cast<int32_t*>(alignedOutput), refOutput,
                  //numberOfVectors * outputVolume);
}

