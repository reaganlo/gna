/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "../test-gna-api/test-gna-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-api.h"

#include <list>

class Gna2OperationHolder
{
    static const int MaxTensorsParams = 6;
    Gna2Tensor tensors[MaxTensorsParams] = {};
    union Gna2Parameter {
        Gna2Shape shape;
        uint32_t uint32;
    } parameters[MaxTensorsParams] = {};
    Gna2Tensor* pTensors[MaxTensorsParams];
    void* pParams[MaxTensorsParams];

    Gna2Operation operation;

    void Init()
    {
        for (int i = 0; i < MaxTensorsParams; i++)
        {
            pTensors[i] = tensors + i;
            pParams[i] = parameters + i;
        }
        operation = { Gna2OperationTypeNone, const_cast<const Gna2Tensor**>(pTensors), 0, pParams, 0 };
    }
public:

    Gna2OperationHolder()
    {
        Init();
    }

    Gna2OperationHolder(const Gna2OperationHolder& from)
    {
        Init();
        for (int i = 0; i < MaxTensorsParams; i++)
        {
            tensors[i] = from.tensors[i];
            parameters[i] = from.parameters[i];
        }
        operation.Type = from.operation.Type;
        operation.NumberOfOperands = from.operation.NumberOfOperands;
        operation.NumberOfParameters = from.operation.NumberOfParameters;
    }

    Gna2Operation GetOperation() const
    {
        return operation;
    }

    void InitCopy(uint32_t rows, uint32_t inputColumns, uint32_t copyColumns, uint32_t outputColumns, void * input, void * output)
    {
        operation.Type = Gna2OperationTypeCopy;
        operation.NumberOfParameters = 1;
        operation.NumberOfOperands = 2;

        parameters[0].shape = Gna2ShapeInit2D(rows, copyColumns);
        tensors[0] = Gna2TensorInit2D(rows, inputColumns, Gna2DataTypeInt16, input);
        tensors[1] = Gna2TensorInit2D(rows, outputColumns, Gna2DataTypeInt16, output);
    }
    void InitDiagonal(uint32_t inputs, uint32_t batches, void * input, void * output, void * weight, void * bias)
    {
        operation.Type = Gna2OperationTypeElementWiseAffine;
        operation.NumberOfOperands = 4;

        tensors[0] = Gna2TensorInit2D(inputs, batches, Gna2DataTypeInt16, input);
        tensors[1] = Gna2TensorInit2D(inputs, batches, Gna2DataTypeInt32, output);
        tensors[2] = Gna2TensorInit1D(inputs, Gna2DataTypeInt16, weight);
        tensors[3] = Gna2TensorInit1D(inputs, Gna2DataTypeInt32, bias);
    }

    void InitDiagonalPwl(uint32_t inputs, uint32_t batches, uint32_t pwlSegments, void * input, void * output, void * weight, void * bias, void * pwl)
    {
        InitDiagonal(inputs, batches, input, output, weight, bias);
        operation.NumberOfOperands = 5;
        tensors[4] = Gna2TensorInit1D(pwlSegments, Gna2DataTypePwlSegment, pwl);
    }
};

class TestModelError : public TestGnaApiEx
{
protected:
    const uint32_t MemoryToUse = 8192;
    static const uint32_t MaxNumberOfLayers = 8192;
    Gna2Operation gnaOperations[MaxNumberOfLayers] = {};
    Gna2Model gnaModel{ 1, gnaOperations };
    std::list<Gna2OperationHolder> createdOperations;
    bool expectValueMatches = true;

    uint32_t modelId;
    Gna2ModelError lastError;
    void * gnaMemory = nullptr;
    Gna2ModelError e = GetCleanedError();
    void SetUp() override
    {
        TestGnaApiEx::SetUp();
        uint32_t grantedMemory;
        const auto status = Gna2MemoryAlloc(MemoryToUse, &grantedMemory, &gnaMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        ASSERT_EQ(grantedMemory, MemoryToUse);
        ASSERT_NE(gnaMemory, nullptr);
        WithOperations({ SimpleCopy, SimpleCopy, SimpleCopy });
    }

    void TearDown() override
    {
        TestGnaApiEx::TearDown();
        const auto status = Gna2MemoryFree(gnaMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        gnaMemory = nullptr;
    }

    Gna2Operation CreateSimpleCopy()
    {
        createdOperations.emplace_back();
        createdOperations.back().InitCopy(8, 8, 8, 8, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleDiagonal()
    {
        createdOperations.emplace_back();
        createdOperations.back().InitDiagonal(32, 8, gnaMemory, gnaMemory, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleDiagonalPwl()
    {
        createdOperations.emplace_back();
        createdOperations.back().InitDiagonalPwl(32, 8, 4, gnaMemory, gnaMemory, gnaMemory, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    void expectEquivalent(const Gna2ModelError& refError)
    {
        EXPECT_EQ(lastError.Source.Type, refError.Source.Type);
        EXPECT_EQ(lastError.Source.OperationIndex, refError.Source.OperationIndex);
        EXPECT_EQ(lastError.Source.OperandIndex, refError.Source.OperandIndex);
        EXPECT_EQ(lastError.Source.ParameterIndex, refError.Source.ParameterIndex);
        for (auto i = 0; i < GNA2_MODEL_ITEM_NUMBER_OF_PROPERTIES; i++)
        {
            EXPECT_EQ(lastError.Source.Properties[i], refError.Source.Properties[i]);
        }
        EXPECT_EQ(lastError.Source.ShapeDimensionIndex, refError.Source.ShapeDimensionIndex);

        EXPECT_EQ(lastError.Reason, refError.Reason);
        if (expectValueMatches)
        {
            EXPECT_EQ(lastError.Value, refError.Value);
        }
    }
    void expectModelError(const Gna2ModelError& refError)
    {
        auto status = Gna2ModelCreate(DeviceIndex, &gnaModel, &modelId);
        EXPECT_EQ(status, Gna2StatusModelConfigurationInvalid);
        status = Gna2ModelGetLastError(&lastError);
        EXPECT_EQ(status, Gna2StatusSuccess);

        expectEquivalent(refError);
    }

    void expectModelError(const Gna2ErrorType reason)
    {
        e.Reason = reason;
        expectModelError(e);
    }

    void withNumberOfOperations(uint32_t numberOfOperations)
    {
        gnaModel.NumberOfOperations = numberOfOperations;
    }

    static Gna2ModelError GetCleanedError()
    {
        Gna2ModelError e = {};
        e.Reason = Gna2ErrorTypeOther;
        e.Value = 0;
        e.Source.Type = Gna2ItemTypeInternal;
        e.Source.OperationIndex = GNA2_DISABLED;
        e.Source.OperandIndex = GNA2_DISABLED;
        e.Source.ParameterIndex = GNA2_DISABLED;
        e.Source.ShapeDimensionIndex = GNA2_DISABLED;
        for (unsigned i = 0; i < sizeof(e.Source.Properties) / sizeof(e.Source.Properties[0]); i++)
        {
            e.Source.Properties[0] = GNA2_DISABLED;
        }
        return e;
    }

    template<class T>
    void update(const Gna2ItemType what, const T newValue)
    {
        e.Source.Type = what;
        e.Value = (int64_t)(newValue);
        if( what == Gna2ItemTypeShapeDimensions && e.Source.OperandIndex >=0 )
        {
            const_cast<uint32_t&>(
                gnaOperations[e.Source.OperationIndex].
                Operands[e.Source.OperandIndex]->Shape.
                Dimensions[e.Source.ShapeDimensionIndex]) = static_cast<uint32_t>(e.Value);
        }
        else if(what == Gna2ItemTypeOperationType)
        {
            gnaOperations[e.Source.OperationIndex].Type = reinterpret_cast<const Gna2OperationType&>(e.Value);
        }
        else if (what == Gna2ItemTypeOperandType)
        {
            const_cast<Gna2DataType&>(
                gnaOperations[e.Source.OperationIndex].
                Operands[e.Source.OperandIndex]->Type) = reinterpret_cast<const Gna2DataType&>(e.Value);
        }
        else if(what == Gna2ItemTypeShapeNumberOfDimensions && e.Source.OperandIndex >= 0)
        {
            auto& shape = const_cast<Gna2Shape&>(
                gnaOperations[e.Source.OperationIndex].
                Operands[e.Source.OperandIndex]->Shape);
            shape.NumberOfDimensions = static_cast<uint32_t>(e.Value);
        }
        else if (what == Gna2ItemTypeOperandData)
        {
            const_cast<void*&>(
                gnaOperations[e.Source.OperationIndex].
                Operands[e.Source.OperandIndex]->Data) = reinterpret_cast<void*>(e.Value);
        }
    }

    void WrongShapeDimensions(int32_t operationIndex,
        int32_t operandIndex,
        int32_t shapeDimensionIndex,
        int badValue,
        Gna2ErrorType errorType);

    void ExpectOperandDataError(int32_t operationIndex, const uint32_t operandIndex,
        void * badPointer = nullptr, Gna2ErrorType errorType = Gna2ErrorTypeNullNotAllowed);

    typedef Gna2Operation(::TestModelError::*OperationCreationFunction)();
    const OperationCreationFunction SimpleCopy = &TestModelError::CreateSimpleCopy;
    const OperationCreationFunction SimpleDiagonal = &TestModelError::CreateSimpleDiagonal;
    const OperationCreationFunction SimpleDiagonalPwl = &TestModelError::CreateSimpleDiagonalPwl;

    void WithOperations(std::vector< OperationCreationFunction > operations);

};
