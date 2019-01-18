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
    static SampleModelForGnaSelfTest GetDefault();
    size_t GetWeightsByteSize() const
    {
        return byteSize(weights);
    }
    size_t GetBiasesByteSize() const
    {
        return byteSize(biases);
    }
    size_t GetInputsByteSize() const
    {
        return byteSize(inputs);
    }
    size_t GetRefScoresByteSize() const
    {
        return byteSize(refScores);
    }
    void CopyWeights(void * dest) const { arrayCopy(dest,weights); }
    void CopyBiases(void * dest) const { arrayCopy(dest, biases); }
    void CopyInputs(void * dest) const { arrayCopy(dest,inputs); }
    int32_t GetRefScore(size_t idx) const { return refScores[idx]; }
private:
    SampleModelForGnaSelfTest() = default;

    // sample weight matrix (8 rows, 16 cols)
    // in case of affine layer this is the left operand of matrix mul
    std::array<int16_t, 8 * 16> weights;
    // sample input matrix (16 rows, 4 cols), consists of 4 input vectors (grouping of 4 is used)
    // in case of affine layer this is the right operand of matrix mul
    std::array<int16_t, 16 * 4> inputs;
    // sample bias vector, will get added to each of the four output vectors
    std::array<int32_t, 8> biases;

    std::array<int32_t, 8 * 4> refScores;
    template<typename Array>
    static void arrayCopy(void * dest, Array& a)
    {
        memcpy(dest, a.data(), byteSize(a));
    }
    template<typename Array>
    static size_t byteSize(const Array& a)
    {
        return a.size() * sizeof(typename Array::value_type);
    }
};
