/*
 INTEL CONFIDENTIAL
 Copyright 2017-2020 Intel Corporation.

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

#include "ChainModel.h"
#include "ModelUtilities.h"

#include <map>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

ChainModel& ChainModel::Affine(bool weights2B, bool pwlEnabled, bool activeListEnabled)
{
    // TODO: active list
    UNREFERENCED_PARAMETER(activeListEnabled);

    if (locked)
    {
        throw;
    }
    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitAffineEx(inVecSz, outVecSz, groupingNum, nullptr, nullptr,
        nullptr, weights2B ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        nullptr, weights2B ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias);

    if (pwlEnabled)
    {
        operationHolder.AddPwl(64, nullptr, Gna2DataTypeInt16);
    }

    modelSize += ModelUtilities::CalculateDnnSize(groupingNum, inVecSz, outVecSz,
        static_cast<uint32_t>(weights2B ? sizeof(int16_t) : sizeof(int8_t)),
        pwlEnabled ? nSegments : 0);

    operations.push_back(operationHolder.GetOperation());
    return *this;
}

ChainModel& ChainModel::Diagonal(bool weights2B, bool pwlEnabled)
{
    if (locked)
    {
        throw;
    }

    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitDiagonalEx(inVecSz, outVecSz, groupingNum, nullptr, nullptr,
        nullptr, weights2B ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        nullptr, weights2B ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias);

    if (pwlEnabled)
    {
        operationHolder.AddPwl(64, nullptr, Gna2DataTypeInt16);
    }

    modelSize += ModelUtilities::CalculateDnnSize(groupingNum, inVecSz, outVecSz,
        static_cast<uint32_t>(weights2B ? sizeof(int16_t) : sizeof(int8_t)),
        pwlEnabled ? nSegments : 0);
    operations.push_back(operationHolder.GetOperation());
    return *this;
}

ChainModel& ChainModel::Multibias(bool weights2B, bool pwlEnabled)
{
    if (locked)
    {
        throw;
    }

    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitAffineEx(inVecSz, outVecSz, groupingNum, nullptr, nullptr,
        nullptr, weights2B ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        nullptr, Gna2DataTypeInt32);

    operationHolder.AddMBParameters(1, 4);

    if (pwlEnabled)
    {
        operationHolder.AddPwl(64, nullptr, Gna2DataTypeInt16);
    }

    modelSize += ModelUtilities::CalculateMultibiasSize(groupingNum, inVecSz, outVecSz,
        static_cast<uint32_t>(weights2B ? sizeof(int16_t) : sizeof(int8_t)),
        pwlEnabled ? nSegments : 0);
    operations.push_back(operationHolder.GetOperation());
    return *this;
}

ChainModel& ChainModel::Convolution(bool pwlEnabled)
{
    if (locked)
    {
        throw;
    }
    const uint32_t nFilters = 4;
    const uint32_t nFilterCoefficients = 48;
    const uint32_t cnnStride = 48;

    const auto outputsPerFilter = (cnnInVecSz - nFilterCoefficients)
        / (cnnStride) + 1;

    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitCnnLegacy(1, cnnInVecSz, outputsPerFilter, nFilters, nFilterCoefficients, cnnStride,
        nullptr, nullptr,
        nullptr,
        nullptr);

    if (pwlEnabled)
    {
        operationHolder.AddPwl(64, nullptr, Gna2DataTypeInt16);
    }

    modelSize += ModelUtilities::CalculateCnnSize(cnnInVecSz, outputsPerFilter, nFilters, nFilterCoefficients, 64);
    operations.push_back(operationHolder.GetOperation());
    return *this;
}

ChainModel& ChainModel::Pooling(Gna2PoolingMode poolingType)
{
    if (locked)
    {
        throw;
    }
    const auto tmpModelSize = modelSize;
    Convolution(true);

    operationHolders.back().AddPooling(poolingType, 6, 6);
    operations.back() = operationHolders.back().GetOperation();

    const auto outputsPerFilter = operations.back().Operands[1]->Shape.Dimensions[1];
    const uint32_t nFilters = operations.back().Operands[2]->Shape.Dimensions[0];
    const uint32_t nFilterCoefficients = operations.back().Operands[2]->Shape.Dimensions[1];
    modelSize = tmpModelSize;

    modelSize += ModelUtilities::CalculateCnnSize(inVecSz, outputsPerFilter, nFilters, nFilterCoefficients, 64);

    return *this;
}

ChainModel& ChainModel::Recurrent(bool weights2B)
{
    if (locked)
    {
        throw;
    }

    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitRnnEx(groupingNum, inVecSz, rnnOutVecSz, 3, 64,
        nullptr, nullptr,
        nullptr, weights2B ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        nullptr, weights2B ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias,
        nullptr);

    modelSize += ModelUtilities::CalculateRnnSize(groupingNum, inVecSz, outVecSz, weights2B ? 2 : 1, 64);
    operations.push_back(operationHolder.GetOperation());
    return *this;
}

ChainModel& ChainModel::Gmm()
{
    if (locked)
    {
        throw;
    }

    const auto stateCount = 8;
    const auto mixtureComponentCount = 1;

    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitGmmFlat(groupingNum, gmmInVecSz, stateCount, mixtureComponentCount,
        nullptr, nullptr,
        nullptr,
        nullptr, Gna2DataTypeUint16,
        nullptr,
        UINT32_MAX);

    modelSize += ModelUtilities::CalculateGmmSize(mixtureComponentCount,
        groupingNum, inVecSz, stateCount, sizeof(uint16_t));
    operations.push_back(operationHolder.GetOperation());
    GmmCount++;
    return *this;
}

ChainModel& ChainModel::Copy()
{
    if (locked)
    {
        throw;
    }


    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitCopyEx(groupingNum, groupingNum, groupingNum, inVecSz, inVecSz, inVecSz,
        nullptr,
        nullptr);

    modelSize += ModelUtilities::CalculateSimpleSize(groupingNum, inVecSz, outVecSz);
    operations.push_back(operationHolder.GetOperation());
    return *this;
}

ChainModel& ChainModel::Transpose()
{
    if (locked)
    {
        throw;
    }

    operationHolders.push_back({});
    auto & operationHolder = operationHolders.back();
    operationHolder.InitTranspose(groupingNum, inVecSz, Gna2DataTypeInt16, nullptr, nullptr);

    modelSize += ModelUtilities::CalculateSimpleSize(groupingNum, inVecSz, inVecSz);
    operations.push_back(operationHolder.GetOperation());
    return *this;
}

Gna2Model& ChainModel::Setup(uint8_t *pinned_memory)
{
    if (locked)
    {
        throw;
    }
    model.NumberOfOperations = static_cast<uint32_t>(operations.size());
    model.Operations = operations.data();

    locked = true;

    for (auto layerIx = uint32_t{ 0 }; layerIx < model.NumberOfOperations; layerIx++)
    {
        auto& operation = model.Operations[layerIx];
        setup_simple_pointers(operation, pinned_memory);
    }

    return model;
}

uint32_t ChainModel::GetModelSize()
{
    return static_cast<uint32_t>(modelSize);
}

uint16_t ChainModel::GetLayerCount() const
{
    return static_cast<uint16_t>(operations.size());
}

uint32_t GetRoundedSize(const Gna2Tensor& operand, const uint32_t significance)
{
    static const std::map<const Gna2DataType, const uint32_t> dataSizes =
    {
        {Gna2DataTypeNone, 0},
        {Gna2DataTypeBoolean, 1},
        {Gna2DataTypeInt4, 1},
        {Gna2DataTypeInt8, 1},
        {Gna2DataTypeInt16, 2},
        {Gna2DataTypeInt32, 4},
        {Gna2DataTypeInt64, 8},
        {Gna2DataTypeUint4, 1},
        {Gna2DataTypeUint8, 1},
        {Gna2DataTypeUint16, 2},
        {Gna2DataTypeUint32, 4},
        {Gna2DataTypeUint64, 8},
        {Gna2DataTypeCompoundBias, 8},
        {Gna2DataTypePwlSegment, 8},
        {Gna2DataTypeWeightScaleFactor, 8},
    };
    const auto bufferSize = Gna2ShapeGetNumberOfElements(&operand.Shape);

    return Gna2RoundUp(bufferSize * dataSizes.at(operand.Type), significance);
}

uint32_t ChainModel::GetInputBuffersSize() const
{
    if (!locked)
    {
        throw;
    }
    const auto& inputOperand = *model.Operations[0].Operands[0];
    return GetRoundedSize(inputOperand, 64);
}

uint32_t ChainModel::GetOutputBuffersSize() const
{
    if (!locked)
    {
        throw;
    }
    const auto& outputOperand = *model.Operations[model.NumberOfOperations - 1].Operands[1];
    return GetRoundedSize(outputOperand, 1);
}

void ChainModel::setup_simple_pointers(Gna2Operation& operation, uint8_t* &pinned_memory)
{
    for (uint32_t i = 0; i < operation.NumberOfOperands; i++)
    {
        if (operation.Operands[i] == nullptr)
        {
            continue;
        }
        auto& operand = const_cast<Gna2Tensor&>(*operation.Operands[i]);
        if (operand.Mode != Gna2TensorModeDisabled)
        {
            const auto sizeNeeded = GetRoundedSize(operand, 64);
            operand.Data = pinned_memory;
            pinned_memory += sizeNeeded;
        }
    }
}

const uint32_t ChainModel::groupingNum = 4;
const uint32_t ChainModel::inVecSz = 16;
const uint32_t ChainModel::cnnInVecSz = 96;
const uint32_t ChainModel::cnnOutVecSz = 4;
const uint32_t ChainModel::outVecSz = 8;
const uint32_t ChainModel::rnnOutVecSz = 32;
const uint32_t ChainModel::nSegments = 64;
const uint32_t ChainModel::gmmInVecSz = 24;

const int8_t ChainModel::weights_1B[outVecSz * inVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t ChainModel::weights_2B[outVecSz * inVecSz] =
{
    -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
    2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
    0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
    -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
    0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
    2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
};

const int16_t ChainModel::inputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4,
    5, -4, -7,  4,
    0,  7,  1, -7,
    1,  6,  7,  9,
    2, -4,  9,  8,
    -5, -1,  2,  9,
    -8, -8,  8,  1,
    -7,  2, -1, -1,
    -9, -5, -8,  5,
    0, -1,  3,  9,
    0,  8,  1, -2,
    -9,  8,  0, -7,
    -9, -8, -1, -4,
    -3, -7, -2,  3,
    -8,  0,  1,  3,
    -4, -6, -8, -2
};

const int16_t ChainModel::cnnInputs[inVecSz * groupingNum] = {
    -5,  9, -7,  4,
    5, -4, -7,  4,
    0,  7,  1, -7,
    1,  6,  7,  9,
    2, -4,  9,  8,
    -5, -1,  2,  9,
    -8, -8,  8,  1,
    -7,  2, -1, -1,
    -9, -5, -8,  5,
    0, -1,  3,  9,
    0,  8,  1, -2,
    -9,  8,  0, -7,
    -9, -8, -1, -4,
    -3, -7, -2,  3,
    -8,  0,  1,  3,
    -4, -6, -8, -2
};

const int32_t ChainModel::regularBiases[outVecSz*groupingNum] = {
    5, 4, -2, 5,
    -7, -5, 4, -1
};

const Gna2CompoundBias ChainModel::compoundBiases[outVecSz*groupingNum] =
{
    { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
    {-7,1,{0}}, {-5,1,{0}}, {4,1,{0}}, {-1,1,{0}},
};
