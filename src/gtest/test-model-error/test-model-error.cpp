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

TEST_F(TestModelError, ZeroNumberOfOperations)
{
    withNumberOfOperations(0);
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeModelNumberOfOperations;
    e.Reason = Gna2ErrorTypeNotGtZero;
    expectModelError(e);
}

TEST_F(TestModelError, WrongInput)
{
    withNumberOfOperations(3);
    gnaOperations[0] = GetSimpleCopy();
    gnaOperations[1] = GetSimpleCopyWrong();
    gnaOperations[2] = GetSimpleCopy();
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Value = Gna2StatusXnnErrorInputVolume;
    e.Source.OperationIndex = 1;
    expectModelError(e);
}

TEST_F(TestModelError, WrongOutput)
{
    withNumberOfOperations(3);
    gnaOperations[0] = CreateSimpleCopy();
    gnaOperations[1] = CreateSimpleCopy();
    gnaOperations[2] = CreateSimpleCopy();
    const_cast<uint32_t&>(gnaOperations[2].Operands[1]->Shape.Dimensions[0]) = 123;
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Value = Gna2StatusXnnErrorOutputVolume;
    e.Source.OperationIndex = 2;
    expectModelError(e);
}

TEST_F(TestModelError, WrongWeights)
{
    withNumberOfOperations(3);
    gnaOperations[0] = CreateSimpleCopy();
    gnaOperations[1] = CreateSimpleCopy();
    gnaOperations[2] = CreateSimpleDiagonal();
    const_cast<uint32_t&>(gnaOperations[2].Operands[2]->Shape.Dimensions[0]) = 123;
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Value = Gna2StatusXnnErrorWeightVolume;
    e.Source.OperationIndex = 2;
    expectModelError(e);
}

TEST_F(TestModelError, WrongWeightDataType)
{
    withNumberOfOperations(3);
    gnaOperations[0] = CreateSimpleCopy();
    gnaOperations[1] = CreateSimpleCopy();
    gnaOperations[2] = CreateSimpleDiagonal();
    const_cast<Gna2DataType&>(gnaOperations[2].Operands[2]->Type ) = Gna2DataTypeInt32;
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Value = Gna2StatusXnnErrorWeightBytes;
    e.Source.OperationIndex = 2;
    expectModelError(e);
}

TEST_F(TestModelError, DISABLED_TooSmalWeightVolume)
{
    withNumberOfOperations(3);
    gnaOperations[0] = CreateSimpleCopy();
    gnaOperations[1] = CreateSimpleCopy();
    gnaOperations[2] = CreateSimpleDiagonal();
    const_cast<uint32_t&>(gnaOperations[2].Operands[2]->Shape.Dimensions[0]) = 16;
    Gna2ModelError e = GetCleanedError();
    e.Source.Type = Gna2ItemTypeInternal;
    e.Value = Gna2StatusXnnErrorWeightVolume;
    e.Source.OperationIndex = 2;
    expectModelError(e);
}
