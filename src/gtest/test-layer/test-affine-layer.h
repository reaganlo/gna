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

#pragma once

#include "test-layer.h"

#include "HardwareCapabilities.h"
#include "ModelWrapper.h"
#include "Validator.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace GNA;

class TestAffineLayer : public TestLayer
{
public:
    TestAffineLayer();
    ~TestAffineLayer();
    TestAffineLayer(const TestAffineLayer& rhs) = delete;

    static constexpr uint16_t numberOfVectors = 4;
    static constexpr uint16_t inputVolume = 16;
    static constexpr uint16_t outputVolume = 8;
    static constexpr uint16_t multibiasVectorCount = 4;
    static constexpr uint16_t numberOfSegments = 64;

    template<typename T>
    static const T input[numberOfVectors * inputVolume];

    template<typename T>
    static const T weight[outputVolume * inputVolume];

    template<typename T>
    static const T bias[outputVolume];

    static const int32_t multibias[outputVolume * multibiasVectorCount];

    template<typename T>
    static const T refOutput[numberOfVectors * outputVolume];

    template<typename T>
    static const T refOutputRecurrent[numberOfVectors * outputVolume];

    Gna2Operation operation = {};
    Gna2Tensor inputTensor = {};
    Gna2Tensor outputTensor = {};
    Gna2Tensor weightTensor = {};
    Gna2Tensor biasTensor = {};
    void *alignedInput = nullptr;
    void *alignedWeight = nullptr;
    void *alignedBias = nullptr;
    void *alignedMultibias = nullptr;
    void *alignedOutput = nullptr;
    void *alignedIntermediateOutput = nullptr;
    intel_pwl_segment_t *alignedPwlSegments = nullptr;

protected:
    template<typename InputType, typename WeightType, typename BiasType>
    void RunAffineTest(Gna2DataType inputType, Gna2DataType weightType, Gna2DataType biasType)
    {
        PrepareAffineTest<InputType, WeightType, BiasType>(inputType, weightType, biasType);

        ExecuteAffineTest();
    }

    template<typename InputType, typename WeightType, typename BiasType>
    void PrepareAffineTest(Gna2DataType inputType, Gna2DataType weightType, Gna2DataType biasType)
    {
        inputTensor = PrepareInput<InputType>(inputType);

        outputTensor = PrepareOutput(Gna2DataTypeInt32);

        weightTensor = PrepareWeight<WeightType>(weightType);

        biasTensor = PrepareBias<BiasType>(biasType);


        auto status = Gna2OperationInitFullyConnectedAffine(&operation, GnaMalloc,
            &inputTensor, &outputTensor, &weightTensor, &biasTensor, nullptr);
        ASSERT_EQ(status, Gna2StatusSuccess);
    }

    template<typename T>
    const Gna2Tensor PrepareInput(Gna2DataType DataType = Gna2DataTypeInt16) const
    {
        auto tensor = PrepareTensor(DataType, alignedInput, "HW", inputVolume, numberOfVectors);
        memcpy_s(alignedInput, sizeof(int16_t) * numberOfVectors * inputVolume, input<T>, sizeof(input<T>));
        return tensor;
    }

    const Gna2Tensor PrepareOutput(Gna2DataType DataType = Gna2DataTypeInt16) const
    {
        auto tensor = PrepareTensor(DataType, alignedOutput, "HW", outputVolume, numberOfVectors);
        return tensor;
    }

    template<typename WeightType>
    const Gna2Tensor PrepareWeight(Gna2DataType DataType = Gna2DataTypeInt16) const
    {
        auto tensor = PrepareTensor(DataType, alignedWeight, "HW", outputVolume, inputVolume);
        memcpy_s(alignedWeight, sizeof(int16_t) * inputVolume * outputVolume, weight<WeightType>, sizeof(weight<WeightType>));
        return tensor;
    }

    template<typename BiasType>
    const Gna2Tensor PrepareBias(Gna2DataType DataType = Gna2DataTypeInt16) const
    {
        auto tensor = PrepareTensor(DataType, alignedBias, "H", outputVolume);
        memcpy_s(alignedBias, sizeof(int32_t) * outputVolume, bias<BiasType>, sizeof(bias<BiasType>));
        return tensor;
    }

    template<typename ... T>
    const Gna2Tensor PrepareTensor(enum Gna2DataType type, void * data, std::string layout, T ... dimensions) const
    {
        auto tensor = ModelWrapper::TensorInit(type, Gna2TensorModeDefault, data, dimensions...);
        strncpy_s(tensor.Layout, GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS, layout.c_str(), layout.size());
        return tensor;
    }

    void ExecuteAffineTest() const;
};
