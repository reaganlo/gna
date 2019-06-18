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

#include "test-gna-api.h"

#include "gna-api.h"
#include "../../gna-api/gna2-model-api.h"

#include "Macros.h"

#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>

class TestGnaModelApi : public TestGnaApi
{
protected:
    static void * Allocator(uint32_t size)
    {
        return malloc(size);
    }
    static void Free(void * ptr)
    {
        return free(ptr);
    }
    static void * InvalidAllocator(uint32_t size)
    {
        UNREFERENCED_PARAMETER(size);
        return nullptr;
    }
};

class TestGnaOperationInitApi : public TestGnaModelApi
{
protected:
    //Operands for Gna2InitOperation???() testing
    Gna2Tensor input{};
    Gna2Tensor output{};
    Gna2Tensor weights{};
    Gna2Tensor filters{};
    Gna2Tensor biases{};
    Gna2Tensor activation{};

    //Parameters for Gna2InitOperation???() testing
    Gna2Shape convolutionStride{};
    Gna2Shape poolingStride{};
    Gna2Shape poolingWindow{};
    Gna2Shape zeroPadding{};
    Gna2BiasMode biasMode{};
    Gna2PoolingMode poolingMode{};

    void ExpectEqual(const Gna2Operation& l, const Gna2Operation& r) const
    {
        EXPECT_EQ(l.Type, r.Type);
        EXPECT_EQ(l.Operands, r.Operands);
        EXPECT_EQ(l.NumberOfOperands, r.NumberOfOperands);
        EXPECT_EQ(l.Parameters, r.Parameters);
        EXPECT_EQ(l.NumberOfParameters, r.NumberOfParameters);
    }

    template<class M, class V, class ... T>
    void ExpectTableFilledBy(M pointerTable, V tableSize, T ... expectedValues)
    {
        EXPECT_EQ(tableSize, sizeof...(expectedValues));
        ASSERT_NE(pointerTable, nullptr);
        for (auto expectedValue : std::initializer_list<const void*>{ expectedValues... })
        {
            EXPECT_EQ(*pointerTable++, expectedValue);
        }
    }
};

class TestGnaShapeApi : public TestGnaApi
{

public:
    template<typename ... T>
    static void InitTest(const Gna2Shape& shape, T... dimensions)
    {
        const auto dimensionList = std::vector<uint32_t>({static_cast<uint32_t>(dimensions)...});
        const auto size = dimensionList.size();
        ASSERT_EQ(shape.NumberOfDimensions, static_cast<uint32_t>(size));

        auto dimIter = dimensionList.begin();
        uint32_t i = 0;
        for (; i < size; ++dimIter, i++)
        {
            ASSERT_EQ(shape.Dimensions[i], *dimIter);
        }
        for (; i < static_cast<uint32_t>(GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS); i++)
        {
            ASSERT_EQ(shape.Dimensions[i], static_cast<uint32_t>(0));
        }
    }
};

class TestGnaTensorApi : public TestGnaApi
{
protected:
    template<typename ... T>
    static void InitTest(const Gna2Tensor& tensor, enum Gna2DataType type,
        void * data, char const* layout, T... dimensions)
    {
        TestGnaShapeApi::InitTest(tensor.Shape, dimensions...);
        ASSERT_EQ(tensor.Data, data);
        ASSERT_STREQ(tensor.Layout, layout);
        ASSERT_EQ(tensor.Mode, Gna2TensorModeDefault);
        ASSERT_EQ(tensor.Type, type);
    }

    int16_t data = 0;
};

TEST_F(TestGnaApi, allocateMemory)
{
    uint32_t sizeRequested = 47;
    uint32_t sizeGranted = 0;
    void * mem = nullptr;
    gna_status_t status;
    status = GnaAlloc(sizeRequested, &sizeGranted, &mem);
    EXPECT_LE(sizeRequested, sizeGranted);
    ASSERT_EQ(status, GNA_SUCCESS);
    GnaFree(mem);
}

TEST_F(TestGnaModelApi, Gna2DataTypeGetSizeSuccesfull)
{
    auto size = Gna2DataTypeGetSize(Gna2DataTypeNone);
    ASSERT_EQ(size, static_cast<uint32_t>(0));
    size = Gna2DataTypeGetSize(Gna2DataTypeBoolean);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt4);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt8);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt16);
    ASSERT_EQ(size, static_cast<uint32_t>(2));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt32);
    ASSERT_EQ(size, static_cast<uint32_t>(4));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt64);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint4);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint8);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint16);
    ASSERT_EQ(size, static_cast<uint32_t>(2));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint32);
    ASSERT_EQ(size, static_cast<uint32_t>(4));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint64);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypeCompoundBias);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypePwlSegment);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypeWeightScaleFactor);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
}

TEST_F(TestGnaModelApi, Gna2DataTypeGetSizeIncorrectType)
{
    auto size = Gna2DataTypeGetSize(static_cast<Gna2DataType>(Gna2DataTypeInt8 - 100));
    ASSERT_EQ(size, static_cast<uint32_t>(GNA2_NOT_SUPPORTED));
}

TEST_F(TestGnaShapeApi, Gna2ShapeGetNumberOfElementsSuccessfull)
{
    Gna2Shape shape = {};
    shape.NumberOfDimensions = 2;
    shape.Dimensions[0] = 2;
    shape.Dimensions[1] = 3;
    const auto size = Gna2ShapeGetNumberOfElements(&shape);
    ASSERT_EQ(size, static_cast<uint32_t>(6));
}

TEST_F(TestGnaShapeApi, Gna2ShapeGetNumberOfElementsNullShape)
{
    const auto size = Gna2ShapeGetNumberOfElements(nullptr);
    ASSERT_EQ(size, static_cast<uint32_t>(GNA2_NOT_SUPPORTED));
}

TEST_F(TestGnaShapeApi, Gna2ShapeGetNumberOfElementsZero)
{
    Gna2Shape shape = {};
    shape.NumberOfDimensions = 0;
    const auto size = Gna2ShapeGetNumberOfElements(&shape);
    ASSERT_EQ(size, static_cast<uint32_t>(0));
}

TEST_F(TestGnaShapeApi, Gna2ShapeGetNumberOfElementsInvalidDimensions)
{
    Gna2Shape shape = {};
    shape.NumberOfDimensions = 0;
    shape.Dimensions[0] = 2;
    const auto size = Gna2ShapeGetNumberOfElements(&shape);
    ASSERT_EQ(size, static_cast<uint32_t>(0));
}

TEST_F(TestGnaShapeApi, Gna2ShapeGetNumberOfElementsInvalidDimensions2)
{
    Gna2Shape shape = {};
    shape.NumberOfDimensions = 99;
    shape.Dimensions[0] = 2;
    const auto size = Gna2ShapeGetNumberOfElements(&shape);
    ASSERT_EQ(size, static_cast<uint32_t>(GNA2_NOT_SUPPORTED));
}

TEST_F(TestGnaShapeApi, Gna2ShapeInitScalarSuccessfull)
{
    const auto shape = Gna2ShapeInitScalar();
    InitTest(shape);
}

TEST_F(TestGnaShapeApi, Gna2ShapeInit1DSuccessfull)
{
    const auto shape = Gna2ShapeInit1D(9);
    InitTest(shape, 9);
}

TEST_F(TestGnaShapeApi, Gna2ShapeInit2DSuccessfull)
{
     const auto shape = Gna2ShapeInit2D(9, 13);
    InitTest(shape, 9, 13);
}

TEST_F(TestGnaShapeApi, Gna2ShapeInit3DSuccessfull)
{
     const auto shape = Gna2ShapeInit3D(9, 13, 42);
    InitTest(shape, 9, 13, 42);
}

TEST_F(TestGnaShapeApi, Gna2ShapeInit4DSuccessfull)
{
    const auto shape = Gna2ShapeInit4D(9, 13, 42, 0);
    InitTest(shape, 9, 13, 42, 0);
}

TEST_F(TestGnaTensorApi, Gna2TensorInitDisabledSuccessfull)
{
    const auto tensor = Gna2TensorInitDisabled();
    ASSERT_EQ(tensor.Data, nullptr);
    ASSERT_STREQ(tensor.Layout, "");
    ASSERT_EQ(tensor.Mode, Gna2TensorModeDisabled);
    ASSERT_EQ(tensor.Type, Gna2DataTypeNone);
}

TEST_F(TestGnaTensorApi, Gna2TensorInitScalarSuccessfull)
{
    const auto tensor = Gna2TensorInitScalar(Gna2DataTypeInt16, &data);
    InitTest(tensor, Gna2DataTypeInt16, &data, "S");
}
TEST_F(TestGnaTensorApi, Gna2TensorInit1DSuccessfull)
{
    const auto tensor = Gna2TensorInit1D(9, Gna2DataTypeInt16, &data);
    InitTest(tensor, Gna2DataTypeInt16, &data, "", 9);
}

TEST_F(TestGnaTensorApi, Gna2TensorInit2DSuccessfull)
{
    const auto tensor = Gna2TensorInit2D(9, 13, Gna2DataTypeInt16, &data);
    InitTest(tensor, Gna2DataTypeInt16, &data, "", 9, 13);
}

TEST_F(TestGnaTensorApi, Gna2TensorInit3DSuccessfull)
{
    const auto tensor = Gna2TensorInit3D(9, 13, 42, Gna2DataTypeInt16, &data);
    InitTest(tensor, Gna2DataTypeInt16, &data, "", 9, 13, 42);
}

TEST_F(TestGnaTensorApi, Gna2TensorInit4DSuccessfull)
{
    const auto tensor = Gna2TensorInit4D(9, 13, 42, 0, Gna2DataTypeInt16, &data);
    InitTest(tensor, Gna2DataTypeInt16, &data, "", 9, 13, 42, 0);
}

TEST_F(TestGnaTensorApi, Gna2TensorInit1dInvalidType)
{
    const auto tensor = Gna2TensorInit1D(9, static_cast<Gna2DataType>(Gna2DataTypeWeightScaleFactor + 100), &data);
    InitTest(tensor, Gna2DataTypeNone, nullptr, "");
}

TEST_F(TestGnaModelApi, Gna2ModelCreateNull)
{
    const auto status = GnaModelCreate(0, nullptr, nullptr);
    ASSERT_NE(GNA_SUCCESS, status);
}

TEST_F(TestGnaModelApi, Gna2ModelCreateEmptyModelUnsuccessfull)
{
    uint32_t modelId = 0;
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(0, &model, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ModelCreateSingleCopyLayerSuccesfull)
{
    uint32_t modelId = 0;

    //TODO:3:P1: Check the proper Dimensions order 16,8 vs 8 16
    Gna2Tensor input{
        Gna2Shape{2, { 8, 16 } },
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt16,
        nullptr };

    Gna2Tensor output{ input };
    Gna2Shape copyShape{ 2, { 8, 16 } };
    void * parameters[] = { &copyShape };
    const Gna2Tensor * inout[] = { &input, &output };
    Gna2Operation copyOperation{ Gna2OperationTypeCopy ,
        inout, 2,
        parameters, 1 };

    Gna2Model model = { 1, &copyOperation };

    auto status = Gna2ModelCreate(0, &model, &modelId);
    ASSERT_EQ(status, Gna2StatusSuccess);
    status = Gna2ModelRelease(modelId);
    ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaModelApi, Gna2ModelCreateSingleConvolutionalLayerSuccesfull)
{
    uint32_t modelId = 0;

    Gna2Tensor input{
        Gna2Shape{4, { 1, 8, 6, 1 } },  //CNN2D NHWC
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt16,
        nullptr };
    Gna2Tensor output{
        Gna2Shape{4, { 1, 7, 5, 2 } },  //CNN2D NHWC
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt32,
        nullptr };
    Gna2Tensor filters{
        Gna2Shape{4, { 2, 2, 2, 1 } },  //CNN2D NHWC
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt16,
        nullptr };
    Gna2Tensor bias{
        Gna2Shape{1, { 2 } },  //bias per 2 kernels
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt32,
        nullptr };
    Gna2Tensor activation{
    Gna2Shape{1, { 2 } },  //2 segments
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypePwlSegment,
        nullptr };

    const Gna2Tensor * convolutionTensorSet[] = { &input, &output, &filters, &bias, &activation };

    Gna2Shape convolutionStride = {2, {1,1}};
    auto biasMode = Gna2BiasModeDefault;
    void * parameters[] = { &convolutionStride, &biasMode};

    Gna2Operation convolutionOperation{ Gna2OperationTypeConvolution ,
        convolutionTensorSet, 5,
        parameters, 2 };

    Gna2Model model = { 1, &convolutionOperation };

    auto status = Gna2ModelCreate(0, &model, &modelId);
    ASSERT_EQ(status, Gna2StatusSuccess);

    status= Gna2ModelRelease(modelId);
    ASSERT_EQ(status, Gna2StatusSuccess);

    //Checking optional activation and bias
    for (auto index : { 4, 3 })
    {
        convolutionTensorSet[index] = nullptr;
        status = Gna2ModelCreate(0, &model, &modelId);
        ASSERT_EQ(status, Gna2StatusSuccess);

        status = Gna2ModelRelease(modelId);
        ASSERT_EQ(status, Gna2StatusSuccess);
    }
}

TEST_F(TestGnaModelApi, Gna2ModelCreateNullModel)
{
    uint32_t modelId = 0;
    const auto status = Gna2ModelCreate(0, nullptr, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ModelCreateNullModelId)
{
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(0, &model, nullptr);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ItemTypeModelOperationsodelCreate2InvalidDeviceIndex)
{
    uint32_t modelId = 0;
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(100, &model, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}


TEST_F(TestGnaModelApi, DISABLED_Gna2ModelCreateSingleGMMSuccesfull)
{
    uint32_t modelId = 0;
    uint32_t const batchSize = 1;
    uint32_t const featureVectorLength = 24;
    uint32_t const gmmStates = 1;
    uint32_t const mixtures = 1;
    auto const dataShape = Gna2Shape{3, { gmmStates, mixtures, featureVectorLength }};  //WHD

    Gna2Tensor input{
        Gna2Shape{2, { batchSize, featureVectorLength } },  // HW
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt8,
        nullptr };
    Gna2Tensor output{
        Gna2Shape{2, { batchSize, gmmStates } },  // HW
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeInt32,
        nullptr };
    Gna2Tensor means{
        dataShape,
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeUint8,
        nullptr };
    Gna2Tensor inverseCovariances{
        dataShape,
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeUint8,
        nullptr };
    Gna2Tensor constants{
        Gna2Shape{2, { gmmStates, mixtures }},  //WH
        Gna2TensorModeDefault,
        {'\0'},
        Gna2DataTypeUint32,
        nullptr };

    const Gna2Tensor * tensors[] = { &input, &output, &means, &inverseCovariances, &constants };

    uint32_t maxScore = UINT32_MAX;
    void * parameters[] = { &maxScore };

    Gna2Operation operation{ Gna2OperationTypeGmm ,
        tensors, 5,
        parameters, 1 };

    Gna2Model model = { 1, &operation };

    const auto status = Gna2ModelCreate(0, &model, &modelId);
    ASSERT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestGnaOperationInitApi, Gna2ModelOperationInitSuccessfull)
{
    struct Gna2Operation operation = {};
    const auto type = Gna2OperationTypeCopy;
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    ASSERT_TRUE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaOperationInitApi, Gna2ModelOperationInitNullOperation)
{
    const auto type = Gna2OperationTypeFullyConnectedAffine;
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(nullptr,
        type, userAllocator);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaOperationInitApi, Gna2ModelOperationInitNotZeroed)
{
    struct Gna2Operation operation = {};
    operation.Operands = reinterpret_cast<Gna2Tensor const **>(&operation);
    const struct Gna2Operation operationCopy { operation };
    const auto type = Gna2OperationTypeCopy;
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    EXPECT_NE(status, Gna2StatusSuccess);
    ExpectEqual(operation, operationCopy);
}

TEST_F(TestGnaOperationInitApi, Gna2ModelOperationInitInvalidType)
{
    struct Gna2Operation operation = {};
    const auto type = static_cast<Gna2OperationType>(100);
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}


TEST_F(TestGnaOperationInitApi, Gna2ModelOperationInitNullAllocator)
{
    struct Gna2Operation operation = {};
    const auto type = Gna2OperationTypeCopy;
    const auto status = Gna2ModelOperationInit(&operation,
        type, nullptr);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaOperationInitApi, Gna2ModelOperationInitInvalidAllocator)
{
    struct Gna2Operation operation = {};
    const auto type = Gna2OperationTypeCopy;
    const auto userAllocator = &InvalidAllocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaOperationInitApi, Gna2OperationInitFullyConnectedAffine)
{
    struct Gna2Operation operation = {};
    auto status = Gna2OperationInitFullyConnectedAffine(&operation, Allocator,
        &input, &output, &weights, &biases, &activation);
    ASSERT_EQ(Gna2StatusSuccess, status);
    EXPECT_EQ(operation.Type, Gna2OperationTypeFullyConnectedAffine);
    ExpectTableFilledBy(operation.Operands, operation.NumberOfOperands,
        &input, &output, &weights, &biases, &activation, nullptr);
    ExpectTableFilledBy(operation.Parameters, operation.NumberOfParameters, nullptr, nullptr);

    free(operation.Parameters);
    free(operation.Operands);
}

TEST_F(TestGnaOperationInitApi, Gna2OperationInitConvolution)
{
    struct Gna2Operation operation = {};
    auto status = Gna2OperationInitConvolution(&operation, Allocator,
        &input, &output, &filters, &biases, &activation,
        &convolutionStride, &biasMode);
    ASSERT_EQ(Gna2StatusSuccess, status);
    EXPECT_EQ(operation.Type, Gna2OperationTypeConvolution);
    ExpectTableFilledBy(operation.Operands, operation.NumberOfOperands,
        &input, &output, &filters, &biases, &activation);
    ExpectTableFilledBy(operation.Parameters, operation.NumberOfParameters,
        &convolutionStride, &biasMode, nullptr, nullptr, nullptr, nullptr);

    free(operation.Parameters);
    free(operation.Operands);
}

TEST_F(TestGnaOperationInitApi, Gna2OperationInitOperationsUnsuccesfullNullOperand)
{
    struct Gna2Operation operation = {};
    auto status = Gna2OperationInitFullyConnectedAffine(&operation, Allocator,
        &input, &output, nullptr, &weights, &activation);
    EXPECT_NE(status, Gna2StatusSuccess);
}

TEST_F(TestGnaOperationInitApi, Gna2OperationInitOperationsUnsuccesfullInvalidAllocator)
{
    struct Gna2Operation operation = {};
    auto status = Gna2OperationInitFullyConnectedAffine(&operation, InvalidAllocator,
        &input, &output, &weights, &biases, &activation);
    EXPECT_NE(status, Gna2StatusSuccess);
}

TEST_F(TestGnaOperationInitApi, Gna2OperationInitOperationsUnsuccesfullNullParameter)
{
    struct Gna2Operation operation = {};
    const auto status = Gna2OperationInitConvolutionFused(&operation, Allocator,
        &input, &output, nullptr, &biases, &activation,
        &convolutionStride, &biasMode, &poolingMode, &poolingWindow, &poolingStride, &zeroPadding);
    EXPECT_NE(status, Gna2StatusSuccess);
}

TEST_F(TestGnaOperationInitApi, Gna2OperationInitOperationsSuccesfullNullActivation)
{
    struct Gna2Operation operation = {};
    const auto status = Gna2OperationInitConvolutionFused(&operation, Allocator,
        &input, &output, &filters, &biases, nullptr,
        &convolutionStride, &biasMode, &poolingMode, &poolingWindow, &poolingStride, &zeroPadding);
    EXPECT_EQ(status, Gna2StatusSuccess);
    Free(operation.Operands);
    Free(operation.Parameters);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
