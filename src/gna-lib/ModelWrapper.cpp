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

#include "ModelWrapper.h"

#include "DataMode.h"
#include "gna2-model-impl.h"

#include <vector>

using std::vector;

using namespace GNA;

void ModelWrapper::OperationInit(ApiOperation& operation, const OperationType type,
    const Gna2UserAllocator userAllocator)
{
    Expect::Equal(operation.Type, Gna2OperationTypeNone, Gna2StatusModelConfigurationInvalid);
    Expect::Equal(operation.NumberOfParameters, static_cast<uint32_t>(0), Gna2StatusModelConfigurationInvalid);
    Expect::Equal(operation.NumberOfOperands, static_cast<uint32_t>(0), Gna2StatusModelConfigurationInvalid);
    Expect::Null(operation.Operands);
    Expect::Null(operation.Parameters);

    operation.Type = type;

    operation.NumberOfOperands = GetNumberOfOperands(type);
    operation.Operands = AllocateAndFillZeros<Gna2Tensor const>(userAllocator, operation.NumberOfOperands);

    operation.NumberOfParameters = GetNumberOfParameters(type);
    operation.Parameters = AllocateAndFillZeros<void>(userAllocator, operation.NumberOfParameters);
}

uint32_t ModelWrapper::DataTypeGetSize(DataType type)
{
    const auto dataSize = DataMode::ToSize<uint32_t>(type);
    return dataSize;
}

uint32_t ModelWrapper::ShapeGetNumberOfElements(ApiShape const * shape)
{
    Expect::NotNull(shape);
    const auto shapeImpl = Shape::Create(*shape);
    return shapeImpl.GetNumberOfElements();
}

uint32_t ModelWrapper::GetNumberOfOperands(OperationType operationType)
{
    static std::map<OperationType, uint32_t> numberOfOperands =
    {
        { Gna2OperationTypeCopy, 2 },
        { Gna2OperationTypeConvolution, 5 },
        { Gna2OperationTypeElementWiseAffine, 5 },
        { Gna2OperationTypeFullyConnectedAffine, 6 },
        { Gna2OperationTypeGmm, 5 },
        { Gna2OperationTypeRecurrent, 5 },
        { Gna2OperationTypeTransposition, 2 },
    };
    try
    {
        return numberOfOperands.at(operationType);
    }
    catch (const std::out_of_range &)
    {
        throw GnaException(Gna2StatusModelConfigurationInvalid);
    }
}

uint32_t ModelWrapper::GetNumberOfParameters(OperationType operationType)
{
    static std::map<OperationType, uint32_t> numberOfParameters =
    {
        { Gna2OperationTypeCopy, 1 },
        { Gna2OperationTypeConvolution, 6 },
        { Gna2OperationTypeElementWiseAffine, 0 },
        { Gna2OperationTypeFullyConnectedAffine, 2 },
        { Gna2OperationTypeGmm, 1 },
        { Gna2OperationTypeRecurrent, 1 },
        { Gna2OperationTypeTransposition, 0 },

    };
    try
    {
        return numberOfParameters.at(operationType);
    }
    catch (const std::out_of_range &)
    {
        throw GnaException(Gna2StatusModelConfigurationInvalid);
    }
}
void ModelWrapper::SetLayout(Gna2Tensor& tensor, const char* layout)
{
    snprintf(tensor.Layout, sizeof(tensor.Layout), "%s", layout);
}