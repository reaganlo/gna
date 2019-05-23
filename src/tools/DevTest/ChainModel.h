
/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include "gna-api.h"

#include <memory>
#include <vector>

class ChainModel
{
public:
    ChainModel();
    ChainModel(ChainModel& rhs) = delete;
    ChainModel operator=(ChainModel&& rhs) = delete;
    ~ChainModel();

    ChainModel& Affine(bool weights2B, bool pwlEnabled, bool activeListEnabled);
    ChainModel& Diagonal(bool weights2B, bool pwlEnabled);
    ChainModel& Multibias(bool weights2B, bool pwlEnabled);
    ChainModel& Convolution(bool pwlEnabled);
    ChainModel& Pooling(intel_pool_type_t poolingType);
    ChainModel& Recurrent(bool weights2B);
    ChainModel& Gmm();
    ChainModel& Copy();
    ChainModel& Transpose();

    intel_nnet_type_t& Setup(uint8_t * pinned_memory);

    uint16_t GetLayerCount() const;
    uint32_t GetModelSize();
    uint32_t GetInputBuffersSize();
    uint32_t GetOutputBuffersSize();

    uint16_t GmmCount = 0;

private:
    void setup_dnn_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory);
    void setup_multibias_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory);
    void setup_cnn_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory);
    void setup_rnn_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory);
    void setup_simple_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory);
    void setup_gmm_pointers(intel_nnet_layer_t *layer, uint8_t* &pinned_memory);

    bool locked = false;

    intel_nnet_type_t nnet;
    size_t modelSize = 0;

    std::vector<intel_nnet_layer_t> layers;

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
    static const intel_bias_t regularBiases[];
    static const intel_compound_bias_t compoundBiases[];
};
