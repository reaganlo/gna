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

#pragma once

#include "Gna2OperationHolder.hpp"

#include <list>
#include <vector>

class ChainModel
{
public:
    ChainModel() = default;
    ChainModel(ChainModel& rhs) = delete;
    ChainModel operator=(ChainModel&& rhs) = delete;
    ~ChainModel() = default;

    ChainModel& Affine(bool weights2B, bool pwlEnabled, bool activeListEnabled);
    ChainModel& Diagonal(bool weights2B, bool pwlEnabled);
    ChainModel& Multibias(bool weights2B, bool pwlEnabled);
    ChainModel& Convolution(bool pwlEnabled);
    ChainModel& Pooling(Gna2PoolingMode poolingType);
    ChainModel& Recurrent(bool weights2B);
    ChainModel& Gmm();
    ChainModel& Copy();
    ChainModel& Transpose();

    Gna2Model& Setup(uint8_t * pinned_memory);

    uint16_t GetLayerCount() const;
    uint32_t GetModelSize();
    uint32_t GetInputBuffersSize() const;
    uint32_t GetOutputBuffersSize() const;

    uint16_t GmmCount = 0;

private:
    static void setup_simple_pointers(Gna2Operation& operation, uint8_t* &pinned_memory);

    bool locked = false;

    Gna2Model model;
    size_t modelSize = 0;

    std::vector<Gna2Operation> operations;
    std::list<Gna2OperationHolder> operationHolders;

    static const uint32_t nSegments;
    static const uint32_t groupingNum;
    static const uint32_t inVecSz;
    static const uint32_t outVecSz;
    static const uint32_t rnnOutVecSz;
    static const uint32_t cnnInVecSz;
    static const uint32_t cnnOutVecSz;
    static const uint32_t gmmInVecSz;

    static const int16_t inputs[];
    static const int16_t cnnInputs[];
    static const int16_t weights_2B[];
    static const int8_t weights_1B[];
    static const int32_t regularBiases[];
    static const Gna2CompoundBias compoundBiases[];
};
