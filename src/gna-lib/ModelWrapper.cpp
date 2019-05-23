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

    operation.NumberOfOperands = GetOperationInfo(type, NumberOfOperandsMax);
    operation.Operands = AllocateAndFillZeros<Gna2Tensor const>(userAllocator, operation.NumberOfOperands);

    operation.NumberOfParameters = GetOperationInfo(type, NumberOfParametersMax);
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

inline uint32_t ModelWrapper::GetOperationInfo(OperationType operationType, OperationInfoKey infoType)
{
    static std::map<OperationType, std::map<OperationInfoKey, uint32_t> > metaOperationInfo =
    {
        { Gna2OperationTypeCopy,
            {
                { NumberOfOperandsMax, 2 },
                { NumberOfOperandsRequired, 2 },
                { NumberOfParametersMax, 1 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
                { ParameterIndexCopyShape, 0 },
            }
        },
        { Gna2OperationTypeConvolution,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 4 },
                { NumberOfParametersMax, 6 },
                { NumberOfParametersRequired, 2 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
                { OperandIndexFilter, 2 },
                { OperandIndexBias, 3 },
                { OperandIndexActivation, 4 },
                { ParameterIndexConvolutionStride, 0 },
                { ParameterIndexBiasMode, 1 },
                { ParameterIndexPoolingMode, 2 },
                { ParameterIndexPoolingWindow, 3 },
                { ParameterIndexPoolingStride, 4 },
                { ParameterIndexZeroPadding, 5 },
            }
        },
        { Gna2OperationTypeElementWiseAffine,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 4 },
                { NumberOfParametersMax, 0 },
                { NumberOfParametersRequired, 0 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
                { OperandIndexWeight, 2 },
                { OperandIndexBias, 3 },
                { OperandIndexActivation, 4 },
            }
        },
        { Gna2OperationTypeFullyConnectedAffine,
            {
                { NumberOfOperandsMax, 6 },
                { NumberOfOperandsRequired, 4 },
                { NumberOfParametersMax, 2 },
                { NumberOfParametersRequired, 0 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
                { OperandIndexWeight, 2 },
                { OperandIndexBias, 3 },
                { OperandIndexActivation, 4 },
                { OperandIndexWeightScaleFactors, 5 },
                { ParameterIndexBiasMode, 0 },
                { ParameterIndexBiasVectorIndex, 1 },
            }
        },
        { Gna2OperationTypeGmm,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 3 },
                { NumberOfParametersMax, 1 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
                { OperandIndexMeans, 2 },               //"flat" layout
                { OperandIndexInverseCovariances, 3 },  //"flat" layout
                { OperandIndexConstants, 4 },           //"flat" layout
                { OperandIndexInterleaved, 2 },         //"interleaved" layout
                { ParameterIndexMaximumScore, 0 },
            }
        },
        { Gna2OperationTypeRecurrent,
            {
                { NumberOfOperandsMax, 5 },
                { NumberOfOperandsRequired, 4 },
                { NumberOfParametersMax, 1 },
                { NumberOfParametersRequired, 1 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
                { OperandIndexWeight, 2 },
                { OperandIndexBias, 3 },
                { OperandIndexActivation, 4 },
                { ParameterIndexDelay, 0 },
            }
        },
        { Gna2OperationTypeTransposition,
            {
                { NumberOfOperandsMax, 2 },
                { NumberOfOperandsRequired, 2 },
                { NumberOfParametersMax, 0 },
                { NumberOfParametersRequired, 0 },
                { OperandIndexInput, 0 },
                { OperandIndexOutput, 1 },
            }
        },
    };
    try
    {
        return metaOperationInfo.at(operationType).at(infoType);
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
template<class T>
void ExpectPointerArrayValid(T ** ptr , uint32_t arraySize,
    uint32_t reqNotNull, uint32_t maxSize, Gna2Status error)
{
    Expect::InRange(arraySize, reqNotNull, maxSize, error);
    if (0 == arraySize)
    {
        Expect::True(ptr == nullptr, error);
    }
    else
    {
        Expect::NotNull(ptr, error);
        for (uint32_t i = 0; i < reqNotNull; i++)
        {
            Expect::NotNull(ptr[i], error);
        }
    }
}

void ModelWrapper::ExpectOperationValid(const Gna2Operation & operation)
{
    const auto opRequired = GetOperationInfo(operation.Type, NumberOfOperandsRequired);
    const auto opMax = GetOperationInfo(operation.Type, NumberOfOperandsMax);
    ExpectPointerArrayValid(operation.Operands, operation.NumberOfOperands,
        opRequired, opMax, Gna2StatusModelConfigurationInvalid);

    const auto paramRequired = GetOperationInfo(operation.Type, NumberOfParametersRequired);
    const auto paramMax = GetOperationInfo(operation.Type, NumberOfParametersMax);
    ExpectPointerArrayValid(operation.Parameters, operation.NumberOfParameters,
        paramRequired, paramMax, Gna2StatusModelConfigurationInvalid);
}

GnaComponentType ModelWrapper::OperandIndexToType(uint32_t operandIndex)
{
    if( 0 == operandIndex )
    {
        return InputComponent;
    }
    if ( 1 == operandIndex )
    {
        return OutputComponent;
    }
    // TODO:3: P2 Implement remaining cases
    throw GnaException(Gna2StatusNotImplemented);
}
