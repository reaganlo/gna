//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018-2020 Intel Corporation
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material contains trade secrets and proprietary
// and confidential information of Intel or its suppliers and licensors.
// The Material is protected by worldwide copyright and trade secret laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed in any way
// without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//*****************************************************************************
#include "SampleModelForGnaSelfTest.h"

SampleModelForGnaSelfTest::SampleModelForGnaSelfTest(const ModelConfig& config) :
    ModelConfig(config),
    gnaInput{Gna2TensorInit2D(modelCustomFCInputSize, modelCustomFCGrouping, Gna2DataTypeInt16, nullptr)},
    gnaOutput{ Gna2TensorInit2D(modelCustomFCOutputSize, modelCustomFCGrouping, Gna2DataTypeInt32, nullptr) },
    gnaWeights{ Gna2TensorInit2D(modelCustomFCOutputSize, modelCustomFCInputSize, Gna2DataTypeInt16, nullptr) },
    gnaBiases{ Gna2TensorInit1D(modelCustomFCOutputSize, Gna2DataTypeInt32, nullptr) }
{
    inputs.resize(modelCustomFCInputSize*modelCustomFCGrouping);
    refScores.resize(modelCustomFCOutputSize*modelCustomFCGrouping);
    weights.resize(modelCustomFCInputSize*modelCustomFCOutputSize);
    biases.resize(modelCustomFCOutputSize);
    if(config.defaultFC)
    {
        SetupDefault();
    }
    else
    {
        SetupSimple();
    }
}

Gna2Model SampleModelForGnaSelfTest::GetGnaModel(void * gnaInputPtr, void * gnaOutputPtr, void * gnaWeightPtr, void * gnaBiasPtr)
{
    gnaInput.Data = gnaInputPtr;
    gnaOutput.Data = gnaOutputPtr;
    gnaWeights.Data = gnaWeightPtr;
    gnaBiases.Data = gnaBiasPtr;

    for(unsigned i=0;i<modelCustomFCNumberOfOperations;i++)
    {
        gnaModelOperations.push_back({ Gna2OperationTypeFullyConnectedAffine, gnaOperationOperands, 4, nullptr, 0 });
    }

    return Gna2Model{ static_cast<uint32_t>(gnaModelOperations.size()), gnaModelOperations.data() };
}

int SampleModelForGnaSelfTest::CountErrors() const
{
    int errorCount = 0;
    for (uint32_t j = 0; j < modelCustomFCOutputSize; j++)
    {
        for (uint32_t i = 0; i < modelCustomFCGrouping; i++)
        {
            uint32_t idx = j * modelCustomFCGrouping + i;

            if (static_cast<int32_t*>(gnaOutput.Data)[idx] != GetRefScore(idx))
            {
                errorCount++;
            }
        }
    }
    return errorCount;
}

// in this sample the numbers are random and meaningless
void SampleModelForGnaSelfTest::SetupDefault() {

    weights = {
        -6, -2, -1, -1, -2, 9, 6, 5, 2, 4, -1, 5, -2, -4, 0, 9,
        -8, 8, -4, 6, 5, 3, -7, -9, 7, 0, -4, -1, 1, 7, 6, -6,
        2, -8, 6, 5, -1, -2, 7, 5, -1, 4, 8, 7, -9, -1, 7, 1,
        0, -2, 1, 0, 6, -6, 7, 4, -6, 0, 3, -2, 1, 8, -6, -2,
        -6, -3, 4, -2, -8, -6, 6, 5, 6, -9, -5, -2, -5, -8, -6, -2,
        -7, 0, 6, -3, -1, -6, 4, 1, -4, -5, -3, 7, 9, -9, 9, 9,
        0, -2, 6, -3, 5, -2, -1, -3, -5, 7, 6, 6, -8, 0, -4, 9,
        2, 7, -8, -7, 8, -6, -6, 1, 7, -4, -4, 9, -6, -6, 5, -7 };
    inputs = {
        -5, 9, -7, 4,
        5, -4, -7, 4,
        0, 7, 1, -7,
        1, 6, 7, 9,
        2, -4, 9, 8,
        -5, -1, 2, 9,
        -8, -8, 8, 1,
        -7, 2, -1, -1,
        -9, -5, -8, 5,
        0, -1, 3, 9,
        0, 8, 1, -2,
        -9, 8, 0, -7,
        -9, -8, -1, -4,
        -3, -7, -2, 3,
        -8, 0, 1, 3,
        -4, -6, -8, -2 };
    biases = { 5, 4, -2, 5, -7, -5, 4, -1 };
    refScores = {
        -177, -85, 29, 28,
        96, -173, 25, 252,
        -160, 274, 157, -29,
        48, -60, 158, -29,
        26, -2, -44, -251,
        -173, -70, -1, -323,
        99, 144, 38, -63,
        20, 56, -103, 10 };
}

void SampleModelForGnaSelfTest::SetupSimple()
{
    const int32_t biasValue = 2;
    const int16_t weightValue = 11;
    const int16_t inputValue = 1;
    std::fill(weights.begin(), weights.end(), weightValue);
    std::fill(inputs.begin(), inputs.end(), inputValue);
    std::fill(biases.begin(), biases.end(), biasValue);
    std::fill(refScores.begin(), refScores.end(), static_cast<int32_t>(biasValue + modelCustomFCInputSize*inputValue*weightValue));
}