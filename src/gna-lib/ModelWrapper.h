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

#ifndef __GNA2_MODEL_WRAPPER_H
#define __GNA2_MODEL_WRAPPER_H

#include "gna2-model-impl.h"

#include "Expect.h"
#include "GnaException.h"
#include "Shape.h"
#include "Tensor.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>

namespace GNA
{

enum OperationInfoKey
{
    NumberOfOperandsRequired, //must be passed from user as not null
    NumberOfOperandsMax,
    NumberOfParametersRequired, //must be passed from user as not null
    NumberOfParametersMax,
    OperandIndexInput,
    OperandIndexOutput,
    OperandIndexScratchPad,
    OperandIndexWeight,
    OperandIndexFilter,
    OperandIndexBias,
    OperandIndexActivation,
    OperandIndexWeightScaleFactors,
    OperandIndexMeans,
    OperandIndexInverseCovariances,
    OperandIndexConstants,
    OperandIndexInterleaved,

    ParameterIndexCopyShape,
    ParameterIndexConvolutionStride,
    ParameterIndexBiasMode,
    ParameterIndexPoolingMode,
    ParameterIndexPoolingWindow,
    ParameterIndexPoolingStride,
    ParameterIndexZeroPadding,
    ParameterIndexBiasVectorIndex,
    ParameterIndexMaximumScore,
    ParameterIndexDelay,
};

class ModelWrapper
{
public:
    static void OperationInit(ApiOperation& operation,
        OperationType type, Gna2UserAllocator userAllocator);

    static uint32_t DataTypeGetSize(DataType type);

    static uint32_t ShapeGetNumberOfElements(ApiShape const * shape);

    static ApiShape ShapeInit()
    {
        const auto shape = Shape(GNA_TENSOR_SCALAR);
        return static_cast<ApiShape>(shape);
    }

    template<typename ... T>
    static ApiShape ShapeInit(T ... dimensions)
    {
        const auto shape = Shape(GNA_TENSOR_ORDER_ANY, static_cast<uint32_t>(dimensions)...);
        return static_cast<ApiShape>(shape);
    }

    template<typename ... T>
    static ApiTensor TensorInit(const DataType dataType, const TensorMode tensorMode,
        void const * buffer, T ... dimensions)
    {
        auto const shape = Shape(GNA_TENSOR_ORDER_ANY, static_cast<uint32_t>(dimensions)...);
        auto const tensor = std::make_unique<Tensor>(shape, dataType, tensorMode, buffer);
        return static_cast<ApiTensor>(*tensor);
    }

    // The first numberOfRequired pointers in source must not be nullptr, otherwise exception is thrown
    template<class T, class V>
    static void TryAssign(T ** const destination, const size_t destinationSize,
        uint32_t numberOfRequired, std::initializer_list<V*> source)
    {
        Expect::True(destinationSize >= source.size(),
            Gna2StatusModelConfigurationInvalid);
        Expect::True(numberOfRequired <= source.size(), Gna2StatusModelConfigurationInvalid);
        int i = 0;
        for (const auto& s : source)
        {
            if (0 < numberOfRequired)
            {
                Expect::NotNull(s);
                --numberOfRequired;
            }
            destination[i++] = s;
        }
        std::fill(destination + i, destination + destinationSize, nullptr);
    }

    template<class ... T>
    static void SetOperands(ApiOperation & operation, T ... operands)
    {
        Expect::Equal(operation.NumberOfOperands, GetOperationInfo(operation.Type, NumberOfOperandsMax),
            Gna2StatusModelConfigurationInvalid);
        const auto requiredNotNull = GetOperationInfo(operation.Type, NumberOfOperandsRequired);
        TryAssign(operation.Operands, operation.NumberOfOperands,
            requiredNotNull, {std::forward<T>(operands)...});
    }

    template<class ... T>
    static void SetParameters(ApiOperation & operation, T ... parameters)
    {
        Expect::Equal(operation.NumberOfParameters, GetOperationInfo(operation.Type, NumberOfParametersMax),
            Gna2StatusModelConfigurationInvalid);
        const auto requiredNotNull = GetOperationInfo(operation.Type, NumberOfParametersRequired);
        TryAssign(operation.Parameters, operation.NumberOfParameters,
            requiredNotNull, {static_cast<void*>(parameters)...});
    }

    static void SetLayout(Gna2Tensor& tensor, const char* layout);

    static void ExpectOperationValid(const Gna2Operation& operation);
    static uint32_t GetOperandIndex(GnaComponentType operand);


    static uint32_t GetOperationInfo(OperationType operationType, OperationInfoKey infoType);

    static Gna2Tensor GetOperand(const Gna2Operation & apiOperation, uint32_t operandIndex);

    static Gna2Tensor GetOptionalOperand(const Gna2Operation& apiOperation,
        uint32_t operandIndex, Gna2Tensor defaultTensor);

    static void ExpectParameterAvailable(const Gna2Operation & operation, uint32_t index);

    template<class T>
    static T GetParameter(const Gna2Operation & operation, OperationInfoKey parameter)
    {
        auto const index = GetOperationInfo(operation.Type, parameter);
        return GetParameter<T>(operation, index);
    }

    template<class T>
    static T GetParameter(const Gna2Operation & operation, uint32_t index)
    {
        ExpectParameterAvailable(operation, index);
        return *static_cast<T*> (operation.Parameters[index]);
    }

    template<class T>
    static T GetOptionalParameter(const Gna2Operation& operation, OperationInfoKey parameter, T defaultValue)
    {
        auto const index = GetOperationInfo(operation.Type, parameter);
        return GetOptionalParameter<T>(operation, index, defaultValue);
    }

    template<class T>
    static T GetOptionalParameter(const Gna2Operation& operation, uint32_t parameterIndex, T defaultValue)
    {
        if(operation.Parameters != nullptr &&
            parameterIndex < operation.NumberOfParameters &&
            nullptr != operation.Parameters[parameterIndex])
        {
            return *static_cast<const T*>(operation.Parameters[parameterIndex]);
        }
        return defaultValue;
    }

private:
    template<typename Type>
    static Type ** AllocateAndFillZeros(const Gna2UserAllocator userAllocator, uint32_t elementCount)
    {
        if (elementCount == 0)
        {
            return nullptr;
        }
        Expect::NotNull((void *)(userAllocator));
        const auto size = static_cast<uint32_t>(sizeof(Type *)) * elementCount;
        const auto memory = userAllocator(size);
        Expect::NotNull(memory, Gna2StatusResourceAllocationError);
        memset(memory, 0, size);
        return static_cast<Type **>(memory);
    }
};

}

#endif //ifndef __GNA2_MODEL_WRAPPER_H
