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
#pragma once

#include "gna2-model-api.h"

#include <cstdint>
#include <vector>
#include <cstring>

struct ModelConfig
{
    bool defaultFC = true;
    uint32_t modelCustomFCNumberOfOperations = 1;
    uint32_t modelCustomFCInputSize = 16;
    uint32_t modelCustomFCOutputSize = 8;
    uint32_t modelCustomFCGrouping = 4;
};

class SampleModelForGnaSelfTest : public ModelConfig
{
public:
    SampleModelForGnaSelfTest(const ModelConfig& config);

    uint32_t GetWeightsByteSize() const
    {
        return byteSize(weights);
    }
    uint32_t GetBiasesByteSize() const
    {
        return byteSize(biases);
    }
    uint32_t GetInputsByteSize() const
    {
        return byteSize(inputs);
    }
    uint32_t GetRefScoresByteSize() const
    {
        return byteSize(refScores);
    }
    void CopyWeights(void * dest) const { arrayCopy(dest,weights); }
    void CopyBiases(void * dest) const { arrayCopy(dest, biases); }
    void CopyInputs(void * dest) const { arrayCopy(dest,inputs); }
    int32_t GetRefScore(uint32_t idx) const { return refScores[idx]; }
    Gna2Model GetGnaModel(void * gnaInputPtr, void * gnaOutputPtr, void * gnaWeightPtr, void * gnaBiasPtr);
    int CountErrors() const;
private:
    void SetupDefault();
    void SetupSimple();

    // weight matrix (out x in)
    std::vector<int16_t> weights;
    // input matrix (in x gr),
    std::vector<int16_t> inputs;
    // bias vector (out)
    std::vector<int32_t> biases;
    // ref scores vector (out x gr)
    std::vector<int32_t> refScores;
    template<typename Array>
    static void arrayCopy(void * dest, Array& a)
    {
        memcpy(dest, a.data(), byteSize(a));
    }
    template<typename Array>
    static uint32_t byteSize(const Array& a)
    {
        return static_cast<uint32_t>(a.size() * sizeof(typename Array::value_type));
    }

    Gna2Tensor gnaInput, gnaOutput, gnaWeights, gnaBiases;
    const Gna2Tensor* gnaOperationOperands[4] = { &gnaInput, &gnaOutput, &gnaWeights, &gnaBiases };
    std::vector<Gna2Operation> gnaModelOperations;
};
