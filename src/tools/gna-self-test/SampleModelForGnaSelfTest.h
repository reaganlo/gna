//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation
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
#include <cstdint>
#include <array>
#include <cstring>

class SampleModelForGnaSelfTest
{
public:
    static constexpr uint32_t NoOfInputs = 16;
    static constexpr uint32_t NoOfOutputs = 8;
    static constexpr uint32_t NoOfGroups = 4; // grouping factor (1-8), specifies how many input vectors are simultaneously run

    static SampleModelForGnaSelfTest GetDefault();
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
private:
    SampleModelForGnaSelfTest() = default;

    // sample weight matrix (8 rows, 16 cols)
    // in case of affine layer this is the left operand of matrix mul
    std::array<int16_t, NoOfOutputs * NoOfInputs> weights;
    // sample input matrix (16 rows, 4 cols), consists of 4 input vectors (grouping of 4 is used)
    // in case of affine layer this is the right operand of matrix mul
    std::array<int16_t, NoOfInputs * NoOfGroups> inputs;
    // sample bias vector, will get added to each of the four output vectors
    std::array<int32_t, NoOfOutputs> biases;

    std::array<int32_t, NoOfOutputs * NoOfGroups> refScores;
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
};
