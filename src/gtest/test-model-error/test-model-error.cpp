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

#include "test-model-error.hpp"
#include "gna2-model-impl.h"

void TestModelError::WithOperations(std::vector< OperationCreationFunction > operations)
{
    withNumberOfOperations(static_cast<uint32_t>(operations.size()));
    for (unsigned i = 0; i < operations.size(); i++)
    {
        gnaOperations[i] = (this->*operations[i])();
    }
}

TEST_F(TestModelError, ZeroNumberOfOperations)
{
    WithOperations({});
    e.Source.Type = Gna2ItemTypeModelNumberOfOperations;
    expectModelError(Gna2ErrorTypeNotGtZero);
}

void TestModelError::WrongShapeDimensions(int32_t operationIndex,
    int32_t operandIndex,
    int32_t shapeDimensionIndex,
    int badValue,
    Gna2ErrorType errorType)
{
    e.Source.OperationIndex = operationIndex;
    e.Source.OperandIndex = operandIndex;
    e.Source.ShapeDimensionIndex = shapeDimensionIndex;
    update(Gna2ItemTypeShapeDimensions, badValue);
    expectModelError(errorType);
}

TEST_F(TestModelError, WrongInput)
{
    WrongShapeDimensions(1, GNA::InputOperandIndex, 0, 9, Gna2ErrorTypeAboveRange);
}

TEST_F(TestModelError, WrongOutput)
{
    WrongShapeDimensions(2, GNA::OutputOperandIndex, 0, 123, Gna2ErrorTypeAboveRange);
}

TEST_F(TestModelError, WrongInputMulti)
{
    WrongShapeDimensions(2, GNA::InputOperandIndex, 1, 123, Gna2ErrorTypeNotMultiplicity);
}

TEST_F(TestModelError, WrongWeights)
{
    WithOperations({ SimpleCopy, SimpleCopy, SimpleDiagonal });
    WrongShapeDimensions(2, GNA::WeightOperandIndex, 0, 123, Gna2ErrorTypeNotMultiplicity);
}

TEST_F(TestModelError, WrongWeightDataType)
{
    e.Source.OperationIndex = 2;
    e.Source.OperandIndex = GNA::WeightOperandIndex;
    WithOperations({ SimpleCopy, SimpleCopy, SimpleDiagonal });
    update(Gna2ItemTypeOperandType, Gna2DataTypeInt32);
    expectModelError(Gna2ErrorTypeNotInSet);
}

TEST_F(TestModelError, DISABLED_TooSmalWeightVolume)
{
    WithOperations({ SimpleCopy, SimpleCopy, SimpleDiagonal });
    WrongShapeDimensions(2, GNA::WeightOperandIndex, 0, 16, Gna2ErrorTypeBelowRange);
}

TEST_F(TestModelError, WrongBiasDimNumber)
{
    e.Source.OperationIndex = 2;
    e.Source.OperandIndex = GNA::BiasOperandIndex;
    WithOperations({ SimpleCopy, SimpleCopy,SimpleDiagonal });
    update(Gna2ItemTypeShapeNumberOfDimensions, 2);
    expectModelError(Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongOperationType)
{
    e.Source.OperationIndex = 1;
    WithOperations({ SimpleDiagonal, SimpleCopy, SimpleDiagonal });
    update(Gna2ItemTypeOperationType, 123);
    expectModelError(Gna2ErrorTypeNotInSet);
}

TEST_F(TestModelError, WrongPwl)
{
    WithOperations({ SimpleCopy, SimpleDiagonalPwl, SimpleCopy });
    WrongShapeDimensions(1, GNA::PwlOperandIndex, 0, 256, Gna2ErrorTypeAboveRange);
}

void TestModelError::ExpectOperandDataError(int32_t operationIndex, const uint32_t operandIndex, void * badPointer, Gna2ErrorType errorType)
{
    WithOperations({ SimpleDiagonalPwl, SimpleDiagonalPwl, SimpleCopy });
    e.Source.OperationIndex = operationIndex;
    e.Source.OperandIndex = operandIndex;
    update(Gna2ItemTypeOperandData, badPointer);
    expectModelError(errorType);
}

TEST_F(TestModelError, WrongBufferAllign)
{
    ExpectOperandDataError(1, GNA::WeightOperandIndex, static_cast<int8_t*>(gnaMemory) + 1, Gna2ErrorTypeNotAligned);
}

TEST_F(TestModelError, DISABLED_WrongBufferNullPwl)
{
    ExpectOperandDataError(1, GNA::PwlOperandIndex);
}

TEST_F(TestModelError, WrongBufferNullBias)
{
    ExpectOperandDataError(1, GNA::BiasOperandIndex);
}

TEST_F(TestModelError, WrongBufferNullWeight)
{
    ExpectOperandDataError(1, GNA::WeightOperandIndex);
}
