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

void TestModelError::WrongShapeParamsDimensions(int32_t operationIndex,
    int32_t parameterIndex,
    int32_t shapeDimensionIndex,
    int badValue,
    Gna2ErrorType errorType)
{
    e.Source.OperationIndex = operationIndex;
    e.Source.ParameterIndex = parameterIndex;
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


TEST_F(TestModelError, DISABLED_OutputCopyNotMultiplicity)
{
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 1, 50, Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, CopyElemsNotMultiplicity)
{
    WithOperations({ SimpleCopy, SimpleCopyBig, SimpleDiagonal });
    WrongShapeParamsDimensions(1, 0, 1, 10, Gna2ErrorTypeNotMultiplicity);
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
    WithOperations({ SimpleCopy, SimpleCopy,SimpleDiagonal });
    e.Source.OperationIndex = 2;
    e.Source.OperandIndex = GNA::BiasOperandIndex;
    update(Gna2ItemTypeShapeNumberOfDimensions, 2);
    expectModelError(Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongActivationNumberOfDimension)
{
    WithOperations({ SimpleDiagonalPwl, SimpleDiagonal, SimpleDiagonal });
    e.Source.OperationIndex = 0;
    e.Source.OperandIndex = GNA::PwlOperandIndex;
    update(Gna2ItemTypeShapeNumberOfDimensions, 5);
    expectModelError(Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongActivationNumberOfDimension0)
{
    WithOperations({ SimpleDiagonalPwl, SimpleDiagonal, SimpleDiagonal });
    e.Source.OperationIndex = 0;
    e.Source.OperandIndex = GNA::PwlOperandIndex;
    update(Gna2ItemTypeShapeNumberOfDimensions, 0);
    expectModelError(Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongActivationDataNull)
{
    WithOperations({ SimpleDiagonalPwl, SimpleDiagonal, SimpleDiagonal });
    e.Source.OperationIndex = 0;
    e.Source.OperandIndex = GNA::PwlOperandIndex;
    update(Gna2ItemTypeOperandData, nullptr);
    expectModelError(Gna2ErrorTypeNullNotAllowed);
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

TEST_F(TestModelError, WrongTransOut)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 0, 9, Gna2ErrorTypeAboveRange);
}

TEST_F(TestModelError, WrongTransOut2)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 0, 7, Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongTransOut3)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 1, 64, Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongTransOut4)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose2, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 0, 7, Gna2ErrorTypeBelowRange);
}

TEST_F(TestModelError, WrongTransOut5)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose2, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 1, 64, Gna2ErrorTypeAboveRange);
}

TEST_F(TestModelError, WrongTransOut6)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose2, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 0, 128, Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongTransOut7)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose2, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 1, 4, Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongDiagMatch)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose2, SimpleDiagonal });
    WrongShapeDimensions(0, GNA::OutputOperandIndex, 0, 128, Gna2ErrorTypeNotEqual);
}

TEST_F(TestModelError, WrongDiagMatch2)
{
    WithOperations({ SimpleDiagonal, SimpleTranspose2, SimpleDiagonal });
    WrongShapeDimensions(2, GNA::OutputOperandIndex, 1, 3, Gna2ErrorTypeNotEqual);
}

// TODO: 4: enable when CNN dispatch mechanism cleaned
TEST_F(TestModelError, DISABLED_WrongDnn2DPoolOut)
{
    WithOperations({ SimpleDiagonal, SimpleCnn2DPool, SimpleDiagonal });
    WrongShapeDimensions(1, GNA::OutputOperandIndex, 1, 32, Gna2ErrorTypeNotEqual);
}
