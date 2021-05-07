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

#include "gna2-model-api.h"

#include <cstdint>
#include <cstdio>

class Gna2OperationHolder
{
    static const int MaxTensorsParams = 6;
    Gna2Tensor tensors[MaxTensorsParams] = {};
    union Gna2Parameter {
        Gna2Shape shape;
        uint32_t uint32;
        Gna2BiasMode biasMode;
        Gna2PoolingMode poolingMode;
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
            tensors[i].Mode = Gna2TensorModeDisabled;
        }
        operation = { Gna2OperationTypeNone, const_cast<const Gna2Tensor**>(pTensors), 0, pParams, 0 };
    }
    template<const int index>
    void SetUpOperand(const Gna2Tensor operand)
    {
        static_assert(index >= 0 && index < MaxTensorsParams, "");
        tensors[index] = operand;
        if(operation.NumberOfOperands <= index)
        {
            operation.NumberOfOperands = index + 1;
        }
    }

    template<const int index, class T>
    void SetUpParameter(const T& parameter)
    {
        static_assert(index >= 0 && index < MaxTensorsParams, "");
        static_assert(sizeof(T) <= sizeof(parameters[0]), "");
        reinterpret_cast<T&>(parameters[index]) = parameter;
        if (operation.NumberOfParameters <= index)
        {
            operation.NumberOfParameters = index + 1;
        }
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

    Gna2OperationHolder& operator=(const Gna2OperationHolder&) = delete;

    Gna2Operation GetOperation() const
    {
        return operation;
    }

    Gna2Operation& Get()
    {
        return operation;
    }

    void InitCopy(uint32_t rows, uint32_t inputColumns, uint32_t copyColumns, uint32_t outputColumns, void * input, void * output)
    {
        operation.Type = Gna2OperationTypeCopy;
        SetUpOperand<0>(Gna2TensorInit2D(rows, inputColumns, Gna2DataTypeInt16, input));
        SetUpOperand<1>(Gna2TensorInit2D(rows, outputColumns, Gna2DataTypeInt16, output));
        SetUpParameter<0>(Gna2ShapeInit2D(rows, copyColumns));
    }

    void InitCopyEx(uint32_t inputRows, uint32_t outputRows, uint32_t copyRows, uint32_t inputColumns, uint32_t copyColumns, uint32_t outputColumns, void * input, void * output)
    {
        InitCopy(copyRows, inputColumns, copyColumns, outputColumns, input, output);
        tensors[0].Shape.Dimensions[0] = inputRows;
        tensors[1].Shape.Dimensions[0] = outputRows;
    }

    void InitDiagonalEx(uint32_t inputs, uint32_t outputs, uint32_t batches, void * input, void * output, void * weight, Gna2DataType weightType, void * bias, Gna2DataType biasType)
    {
        operation.Type = Gna2OperationTypeElementWiseAffine;
        SetUpOperand<0>(Gna2TensorInit2D(inputs, batches, Gna2DataTypeInt16, input));
        SetUpOperand<1>(Gna2TensorInit2D(outputs, batches, Gna2DataTypeInt32, output));
        SetUpOperand<2>(Gna2TensorInit1D(inputs, weightType, weight));
        SetUpOperand<3>(Gna2TensorInit1D(inputs, biasType, bias));
    }

    void InitAffineEx(uint32_t inputs, uint32_t outputs, uint32_t batches,
        void * input, void * output, void * weight, Gna2DataType weightType, void * bias, Gna2DataType biasType)
    {
        InitDiagonalEx(inputs, outputs, batches, input, output, weight, weightType, bias, biasType);
        operation.Type = Gna2OperationTypeFullyConnectedAffine;
        tensors[2].Shape.NumberOfDimensions = 2;
        tensors[2].Shape.Dimensions[0] = outputs;
        tensors[2].Shape.Dimensions[1] = inputs;
    }

    void InitDiagonal(uint32_t inputs, uint32_t batches, void * input, void * output, void * weight, void * bias)
    {
        InitDiagonalEx(inputs, inputs, batches, input, output, weight, Gna2DataTypeInt16, bias, Gna2DataTypeInt32);
    }
    void AddPwl(const uint32_t pwlSegments, void * pwl, const Gna2DataType newOutType = Gna2DataTypeNone)
    {
        SetUpOperand<4>(Gna2TensorInit1D(pwlSegments, Gna2DataTypePwlSegment, pwl));
        if(newOutType != Gna2DataTypeNone)
        {
            tensors[1].Type = newOutType;
        }
    }
    void InitDiagonalPwl(uint32_t inputs, uint32_t batches, uint32_t pwlSegments, void * input, void * output, void * weight, void * bias, void * pwl)
    {
        InitDiagonal(inputs, batches, input, output, weight, bias);
        AddPwl(pwlSegments, pwl);
    }

    void AddMBParameters(const uint32_t biasVectorIndex, const uint32_t biasGroups)
    {
        tensors[3].Shape.NumberOfDimensions = 2;
        tensors[3].Shape.Dimensions[1] = biasGroups;
        SetUpParameter<0>(Gna2BiasModeGrouping);
        SetUpParameter<1>(biasVectorIndex);
    }

    void InitMB(uint32_t inputs, uint32_t outputs, uint32_t batches,
        uint32_t pwlSegments, uint32_t biasIndex, uint32_t biasGrouping,
        void * input, void * output, void * weight, void * bias, void * pwl)
    {
        operation.Type = Gna2OperationTypeFullyConnectedAffine;
        SetUpOperand<0>(Gna2TensorInit2D(inputs, batches, Gna2DataTypeInt16, input));
        SetUpOperand<1>(Gna2TensorInit2D(outputs, batches, Gna2DataTypeInt16, output));
        SetUpOperand<2>(Gna2TensorInit2D(outputs, inputs, Gna2DataTypeInt16, weight));
        SetUpOperand<3>(Gna2TensorInit2D(outputs, biasGrouping, Gna2DataTypeInt32, bias));
        SetUpOperand<4>(Gna2TensorInit1D(pwlSegments, Gna2DataTypePwlSegment, pwl));
        AddMBParameters(biasIndex, biasGrouping);
    }

    void AddWeightScaleFactor(uint32_t outVecSz, void * weight_scales)
    {
        SetUpOperand<5>(Gna2TensorInit1D(outVecSz, Gna2DataTypeWeightScaleFactor, weight_scales));
    }

    void InitRnnEx(const uint32_t inputVectors,
        uint32_t inputs, uint32_t outputs, uint32_t delay, uint32_t pwlSegments,
        void * input, void * output,
        void * weight, Gna2DataType weightType,
        void * bias, Gna2DataType biasType,
        void * pwl)
    {
        operation.Type = Gna2OperationTypeRecurrent;
        SetUpOperand<0>(Gna2TensorInit2D(inputVectors, inputs, Gna2DataTypeInt16, input));
        SetUpOperand<1>(Gna2TensorInit2D(inputVectors, outputs, Gna2DataTypeInt16, output));
        SetUpOperand<2>(Gna2TensorInit2D(outputs, outputs + inputs, weightType, weight));
        SetUpOperand<3>(Gna2TensorInit1D(outputs, biasType, bias));
        SetUpOperand<4>(Gna2TensorInit1D(pwlSegments, Gna2DataTypePwlSegment, pwl));
        SetUpParameter<0>(delay);
    }

    void InitRnn(uint32_t inputs, uint32_t outputs, uint32_t delay, uint32_t pwlSegments, void * input, void * output, void * weight, void * bias, void * pwl)
    {
        const uint32_t inputVectors = 4;
        InitRnnEx(inputVectors, inputs, outputs, delay, pwlSegments,
            input, output,
            weight, Gna2DataTypeInt16,
            bias, Gna2DataTypeInt32,
            pwl);
    }

    uint32_t Cnn2DOut(uint32_t input, uint32_t filter, uint32_t stride, uint32_t poolWin)
    {
        return (input - filter + 1) / stride - poolWin + 1;
    }

    void InitCnn2DPool(uint32_t inputH, uint32_t inputW, uint32_t inputC,
        uint32_t filterN, uint32_t filterH, uint32_t filterW,
        uint32_t strideH, uint32_t strideW,
        uint32_t poolWinH, uint32_t poolWinW,
        Gna2DataType inputAndFilterType, Gna2DataType outputAndBiasType,
        void * input, void * output, void * filters, void * bias,
        const Gna2PoolingMode poolingMode = Gna2PoolingModeMax)
    {
        const uint32_t poolStride = 1;
        operation.Type = Gna2OperationTypeConvolution;
        SetUpOperand<0>(Gna2TensorInit4D(1, inputH, inputW, inputC, inputAndFilterType, input));
        SetUpOperand<1>(Gna2TensorInit4D(1,
            Cnn2DOut(inputH, filterH, strideH, poolWinH),
            Cnn2DOut(inputW, filterW, strideW, poolWinW),
            filterN, outputAndBiasType, output));
        SetUpOperand<2>(Gna2TensorInit4D(filterN, filterH, filterW, inputC, inputAndFilterType, filters));
        SetUpOperand<3>(Gna2TensorInit1D(filterN, outputAndBiasType, bias));
        SetUpParameter<0>(Gna2ShapeInit2D(strideH, strideW));
        SetUpParameter<1>(Gna2BiasModeDefault);
        SetUpParameter<2>(poolingMode);
        SetUpParameter<3>(Gna2ShapeInit2D(poolWinH, poolWinW));
        SetUpParameter<4>(Gna2ShapeInit2D(poolStride, poolStride));
    }

    void InitCnnLegacy(const uint32_t groupingNum, const uint32_t inVecSz, const uint32_t outPerFilter,
        const uint32_t nFilters, const uint32_t nFilterCoefficients, const uint32_t convStride,
        void * input, void * output, void * filters, void * biases)
    {
        operation.Type = Gna2OperationTypeConvolution;
        SetUpOperand<0>(Gna2TensorInit2D(groupingNum, inVecSz, Gna2DataTypeInt16, input));
        SetUpOperand<1>(Gna2TensorInit3D(groupingNum, outPerFilter, nFilters, Gna2DataTypeInt32, output));
        snprintf(tensors[1].Layout, sizeof(tensors[1].Layout), "GNA1");
        SetUpOperand<2>(Gna2TensorInit2D(nFilters, nFilterCoefficients, Gna2DataTypeInt16, filters));
        SetUpOperand<3>(Gna2TensorInit1D(nFilters, Gna2DataTypeInt32, biases));
        SetUpParameter<0>(Gna2ShapeInit1D(convStride));
        SetUpParameter<1>(Gna2BiasModeDefault);
        SetUpParameter<2>(Gna2PoolingModeDisabled);
    }

    void AddPooling(Gna2PoolingMode poolMode, const uint32_t poolWin, const uint32_t poolStride)
    {
        const auto maxNCOE = (tensors[0].Shape.Dimensions[1] - tensors[2].Shape.Dimensions[1]) /
                             (parameters[0].shape.Dimensions[0]) + 1;
        const auto outputsPerFilter = (maxNCOE - 1) / poolStride + 1;
        tensors[1].Shape.Dimensions[1] = outputsPerFilter;
        SetUpParameter<2>(poolMode);
        SetUpParameter<3>(Gna2ShapeInit1D(poolWin));
        SetUpParameter<4>(Gna2ShapeInit1D(poolStride));
    }

    void InitTranspose(uint32_t rows, uint32_t cols, Gna2DataType type, void * input, void * output)
    {
        operation.Type = Gna2OperationTypeTransposition;
        SetUpOperand<0>(Gna2TensorInit2D(rows, cols, type, input));
        SetUpOperand<1>(Gna2TensorInit2D(cols, rows, type, output));
    }

    void InitGmm(uint32_t batchSize, uint32_t featureVectorLength, uint32_t gmmStates, uint32_t mixtures,
        void *input, void *output, void *weight, uint32_t maxScore)
    {
        operation.Type = Gna2OperationTypeGmm;
        SetUpOperand<0>(Gna2TensorInit2D(batchSize, featureVectorLength, Gna2DataTypeUint8, input));
        SetUpOperand<1>(Gna2TensorInit2D(gmmStates, batchSize, Gna2DataTypeUint32, output));
        SetUpOperand<2>(Gna2TensorInit3D(gmmStates, mixtures, featureVectorLength, Gna2DataTypeUint8, weight));
        SetUpParameter<0>(maxScore);
    }

    void InitGmmFlat(uint32_t batchSize, uint32_t featureVectorLength, uint32_t gmmStates, uint32_t mixtures,
        void *input, void *output,
        void *means,
        void *invConv, const Gna2DataType invConvType,
        void *consts,
        uint32_t maxScore)
    {
        InitGmm(batchSize, featureVectorLength, gmmStates, mixtures, input, output, means, maxScore);
        SetUpOperand<3>(Gna2TensorInit3D(gmmStates, mixtures, featureVectorLength, invConvType, invConv));
        SetUpOperand<4>(Gna2TensorInit2D(gmmStates, Gna2RoundUp(mixtures, 2), Gna2DataTypeUint32, consts));
    }
};
