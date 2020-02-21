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
};

class TestModelError : public TestGnaApiEx
{
protected:
    const uint32_t MemoryToUse = 8192;
    static const uint32_t MaxNumberOfLayers = 8192;
    Gna2Operation gnaOperations[MaxNumberOfLayers] = {};
    Gna2Model gnaModel{ 1, gnaOperations };
    std::list<Gna2OperationHolder> createdOperations;

    uint32_t modelId;
    Gna2ModelError lastError;
    void * gnaMemory = nullptr;

    void SetUp() override
    {
        TestGnaApiEx::SetUp();
        uint32_t grantedMemory;
        const auto status = Gna2MemoryAlloc(MemoryToUse, &grantedMemory, &gnaMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        ASSERT_EQ(grantedMemory, MemoryToUse);
        ASSERT_NE(gnaMemory, nullptr);
    }

    void TearDown() override
    {
        TestGnaApiEx::TearDown();
        const auto status = Gna2MemoryFree(gnaMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        gnaMemory = nullptr;
    }

    Gna2Operation GetSimpleCopy()
    {
        static Gna2OperationHolder copy;
        copy.InitCopy(8, 8, 8, 8, gnaMemory, gnaMemory);
        return copy.GetOperation();
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

    Gna2Operation GetSimpleCopyWrong()
    {
        static Gna2OperationHolder copyWrong;
        copyWrong.InitCopy(9, 16, 16, 16, gnaMemory, gnaMemory);
        return copyWrong.GetOperation();
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
        EXPECT_EQ(lastError.Value, refError.Value);
    }
    void expectModelError(const Gna2ModelError& refError)
    {
        auto status = Gna2ModelCreate(DeviceIndex, &gnaModel, &modelId);
        EXPECT_EQ(status, Gna2StatusModelConfigurationInvalid);
        status = Gna2ModelGetLastError(&lastError);
        EXPECT_EQ(status, Gna2StatusSuccess);

        expectEquivalent(refError);
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
            e.Source.Properties[0] = GNA2_DISABLED;
        }
        return e;
    }

};
