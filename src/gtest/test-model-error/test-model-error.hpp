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

#include "Gna2OperationHolder.hpp"

#include "gna2-memory-api.h"
#include "gna2-model-api.h"

#include <list>

class TestModelError : public TestGnaModel
{
protected:
    static const uint32_t MaxNumberOfLayers = 8192;
    Gna2Operation gnaOperations[MaxNumberOfLayers] = {};
    Gna2Model gnaModel{ 1, gnaOperations };
    std::list<Gna2OperationHolder> createdOperations;
    bool expectValueMatches = true;

    uint32_t modelId;
    Gna2ModelError lastError;

    Gna2ModelError e = GetCleanedError();

    void ReAllocateGnaMemory(const uint32_t required)
    {
        ASSERT_NE(gnaMemory, nullptr);
        const auto status = Gna2MemoryFree(gnaMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        gnaMemory = nullptr;
        AllocateGnaMemory(required, gnaMemory);
        WithOperations({});
    }

    void SetUp() override
    {
        TestGnaModel::SetUp();
        WithOperations({ SimpleCopy, SimpleCopy, SimpleCopy });
    }

    Gna2OperationHolder& allocateNewOperation()
    {
        createdOperations.emplace_back();
        return createdOperations.back();
    }

    Gna2Operation CreateSimpleCopy()
    {
        auto& op = allocateNewOperation();
        op.InitCopy(8, 8, 8, 8, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleCopyBig()
    {
        auto& op = allocateNewOperation();
        op.InitCopy(8, 80, 40, 160, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleDiagonal()
    {
        auto& op = allocateNewOperation();
        op.InitDiagonal(32, 8, gnaMemory, gnaMemory, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleDiagonalPwl()
    {
        auto& op = allocateNewOperation();
        op.InitDiagonalPwl(32, 8, 4, gnaMemory, gnaMemory, gnaMemory, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleMB()
    {
        auto& op = allocateNewOperation();
        op.InitMB(32, 8, 4, 7, 1, 6, gnaMemory, gnaMemory, gnaMemory, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleRnn()
    {
        auto& op = allocateNewOperation();
        op.InitRnn(32, 32, 1, 5, gnaMemory, static_cast<uint8_t*>(gnaMemory) + 1024, gnaMemory, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleTranspose()
    {
        auto& op = allocateNewOperation();
        op.InitTranspose(32, 8, Gna2DataTypeInt16, gnaMemory, gnaMemory);
        return op.GetOperation();
    }

    Gna2Operation CreateSimpleTranspose2()
    {
        auto& op = allocateNewOperation();
        op.InitTranspose(6, 32, Gna2DataTypeInt16, gnaMemory, gnaMemory);
        return createdOperations.back().GetOperation();
    }

    Gna2Operation CreateSimpleCnn2DPool()
    {
        auto& op = allocateNewOperation();
        op.InitCnn2DPool(16, 32, 3, 10, 3, 3, 2, 2, 3, 3,
            Gna2DataTypeInt16, Gna2DataTypeInt16,   // TODO: 4: Check whether configuration valid
            gnaMemory, gnaMemory, gnaMemory, gnaMemory);
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
        e.Reason = Gna2ErrorTypeNone;
        e.Value = 0;
        e.Source.Type = Gna2ItemTypeNone;
        e.Source.OperationIndex = GNA2_DISABLED;
        e.Source.OperandIndex = GNA2_DISABLED;
        e.Source.ParameterIndex = GNA2_DISABLED;
        e.Source.ShapeDimensionIndex = GNA2_DISABLED;
        for (unsigned i = 0; i < sizeof(e.Source.Properties) / sizeof(e.Source.Properties[0]); i++)
        {
            e.Source.Properties[i] = GNA2_DISABLED;
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
        else if( what == Gna2ItemTypeShapeDimensions && e.Source.ParameterIndex >= 0 )
        {
            reinterpret_cast<Gna2Shape*>(
                gnaOperations[e.Source.OperationIndex].
                Parameters[e.Source.ParameterIndex])->
                Dimensions[e.Source.ShapeDimensionIndex] = static_cast<uint32_t>(e.Value);
        }
        else if(what == Gna2ItemTypeOperationType)
        {
            gnaModel.Operations[e.Source.OperationIndex].Type = reinterpret_cast<const Gna2OperationType&>(e.Value);
        }
        else if(what == Gna2ItemTypeOperandMode)
        {
            const_cast<Gna2TensorMode&>(
                gnaOperations[e.Source.OperationIndex].
                Operands[e.Source.OperandIndex]->Mode) = reinterpret_cast<const Gna2TensorMode&>(e.Value);
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
        else if (what == Gna2ItemTypeParameter)
        {
            *reinterpret_cast<int64_t*>(const_cast<void*>(
                gnaOperations[e.Source.OperationIndex].
                Parameters[e.Source.ParameterIndex])) = e.Value;
        }
        else if(what == Gna2ItemTypeOperationOperands && e.Source.OperandIndex >= 0)
        {
            const_cast<Gna2Tensor*&>(
                gnaOperations[e.Source.OperationIndex].
                Operands[e.Source.OperandIndex]) = reinterpret_cast<Gna2Tensor * >(e.Value);
        }
    }

    void WrongShapeDimensions(int32_t operationIndex,
        int32_t operandIndex,
        int32_t shapeDimensionIndex,
        int badValue,
        Gna2ErrorType errorType);

    void WrongShapeParamsDimensions(int32_t operationIndex,
        int32_t parameterIndex,
        int32_t shapeDimensionIndex,
        int badValue,
        Gna2ErrorType errorType);

    void WrongType(int32_t operationIndex,
        int32_t operandIndex,
        int32_t badValue,
        Gna2ErrorType errorType);

    void ExpectOperandDataError(int32_t operationIndex, const uint32_t operandIndex,
        void * badPointer = nullptr, Gna2ErrorType errorType = Gna2ErrorTypeNullNotAllowed);

    typedef Gna2Operation(::TestModelError::*OperationCreationFunction)();
    const OperationCreationFunction SimpleCopy = &TestModelError::CreateSimpleCopy;
    const OperationCreationFunction SimpleDiagonal = &TestModelError::CreateSimpleDiagonal;
    const OperationCreationFunction SimpleDiagonalPwl = &TestModelError::CreateSimpleDiagonalPwl;
    const OperationCreationFunction SimpleTranspose = &TestModelError::CreateSimpleTranspose;
    const OperationCreationFunction SimpleTranspose2 = &TestModelError::CreateSimpleTranspose2;
    const OperationCreationFunction SimpleCnn2DPool = &TestModelError::CreateSimpleCnn2DPool;
    const OperationCreationFunction SimpleCopyBig = &TestModelError::CreateSimpleCopyBig;
    const OperationCreationFunction SimpleRnn = &TestModelError::CreateSimpleRnn;
    const OperationCreationFunction SimpleMB = &TestModelError::CreateSimpleMB;
    void WithOperations(std::vector< OperationCreationFunction > operations);

};
