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

#include "IModelSetup.h"

#include <array>
#include <map>
#include <vector>

#include "DeviceController.h"

class SetupSplitModel : public IModelSetup
{
public:
    gna_model_id ModelId(int modelIndex) const override
    {
        return models.at(modelIndex);
    }

    gna_request_cfg_id ConfigId(int modelIndex, int configIndex) const override
    {
        auto modelIdSplit = ModelId(modelIndex);
        return modelsConfigurations.at(modelIdSplit).at(configIndex);
    }

    SetupSplitModel(DeviceController & deviceCtrl, bool weight2B, bool activeListEn, bool pwlEn);

    ~SetupSplitModel();

    void checkReferenceOutput(int modelIndex, int configIndex) const override;

private:
    size_t getFirstModelSize();
    size_t getSecondModelSize();

    void setupSecondAffineLayer(uint8_t* &pinned_mem_ptr);
    void setupFirstAffineLayer(uint8_t* &pinned_mem_ptr);

    void setupInputBuffer(uint8_t* &pinned_memory, int modelIndex, int configIndex);
    void setupOutputBuffer(uint8_t* &pinned_memory, int modelIndex, int configIndex);

    DeviceController & deviceController;

    bool weightsAre2Bytes;
    bool activeListEnabled;
    bool pwlEnabled;

    uint32_t indicesCount;
    uint32_t *indices;

    uint32_t nSegments = 64;

    intel_nnet_type_t& firstNnet = nnet;
    intel_nnet_type_t secondNnet;

    intel_affine_func_t firstAffineFunc;
    intel_affine_func_t secondAffineFunc;

    intel_pwl_func_t firstPwl;
    intel_pwl_func_t secondPwl;

    intel_affine_layer_t firstAffineLayer;
    intel_affine_layer_t secondAffineLayer;

    std::vector<gna_model_id> models;
    std::map<gna_model_id, std::vector<gna_request_cfg_id>> modelsConfigurations;

    std::map<gna_model_id, std::map<gna_request_cfg_id, std::pair<void*, void*>>> configurationBuffers;

    const int8_t weights_1B[outVecSz * inVecSz] =
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

    const int16_t weights_2B[outVecSz * inVecSz] =
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

    const intel_bias_t regularBiases[outVecSz*groupingNum] =
    {
        5, 4, -2, 5, -7, -5, 4, -1
    };

    const  intel_compound_bias_t compoundBiases[outVecSz*groupingNum] =
    {
        { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}},
        {-7,1,{0}}, {-5,1,{0}}, {4,1,{0}}, {-1,1,{0}},
    };

    const uint32_t alIndices[outVecSz / 2]
    {
        0, 2, 4, 7
    };

    static constexpr int diagonalInVecSz = 16;
    static constexpr int diagonalOutVecSz = 16;

    const int8_t diagonal_weights_1B[diagonalInVecSz] =
    {
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    };

    const int16_t diagonal_weights_2B[diagonalInVecSz] =
    {
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    };

    const intel_bias_t diagonalRegularBiases[diagonalOutVecSz] =
    {
        5, 4, -2, 5, -7, -5, 4, -1, 5, 4, -2, 5, -7, -5, 4, -1
    };

    const  intel_compound_bias_t diagonalCompoundBiases[diagonalOutVecSz * groupingNum] =
    {
        { 5,1,{0} }, {4,1,{0}}, {-2,1,{0}}, {5,1,{0}}, {-7,1,{0}}, {-5,1,{0}}, {4,1,{0}}, {-1,1,{0}},
        { 5,1,{ 0 } },{ 4,1,{ 0 } },{ -2,1,{ 0 } },{ 5,1,{ 0 } },{ -7,1,{ 0 } },{ -5,1,{ 0 } },{ 4,1,{ 0 } },{ -1,1,{ 0 } },
    };

    const std::map<uint32_t /*modelIndex*/, const std::map<uint32_t /*configIndex*/,
        const std::array<int16_t, SetupSplitModel::inVecSz * SetupSplitModel::groupingNum>>> inputs
    {{
        {0, {
            {0, {-5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
                  2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
                 -9, -5, -8,  5, 0, -1,  3,  9, 0,  8,  1, -2, -9,  8,  0, -7,
                 -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2}
            },
            {1, { -9, -5, -8,  5,  0, -1,  3,  9,  0,  8,  1, -2, -9,  8,  0, -7,
                  -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2,
                  -5,  9, -7,  4,  5,-4, -7,  4,  0,  7,  1, -7,  1,  6,  7,  9,
                   2,-4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1 }}
        }},
        {1, {
            {0, { -5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
                   2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
                  -9, -5, -8,  5, 0, -1,  3,  9, 0,  8,  1, -2, -9,  8,  0, -7,
                  -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2 }
            },
            {1, { -9, -5, -8,  5,  0, -1,  3,  9,  0,  8,  1, -2, -9,  8,  0, -7,
                  -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2,
                  -5,  9, -7,  4,  5,-4, -7,  4,  0,  7,  1, -7,  1,  6,  7,  9,
                   2,-4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1 }}
        }}
    }};

    const std::map<uint32_t /*configIndex*/,
        const std::array<int32_t, SetupSplitModel::outVecSz * SetupSplitModel::groupingNum>> affineOutputs
    {{
                {0, { -177, -85, 29, 28, 96, -173, 25, 252, -160, 274, 157, -29, 48, -60, 158, -29,
                        26, -2, -44, -251, -173, -70, -1, -323, 99, 144, 38, -63, 20, 56, -103, 10 }},
                {1, { -41, -1,   -47,   24, -15, 3,  159, 118, -170, 132, -56, -158, -51, -28, -9, -31,
                      102,  233, -83, -147,   6, 22, 301,   4,   31, 109, -53,  -85,  15, -85,  5,  98 }}
    }};

    const std::map<uint32_t /*configIndex*/,
        const std::array<int32_t, SetupSplitModel::diagonalOutVecSz * SetupSplitModel::groupingNum>> diagonalOutputs
    {{
        { 0,{ 35, -49,  47, -19,  -6,  12,  18,  -4, -2,  -9,  -3,    5,   4,  -1,  -2,  -4,
        -11,   1, -25, -23, -50, -14,  13,  76, -44, -44,  52,  10, -36,   9,  -6,  -6,
        -13,  -5, -11,  15,   4,   0,  16,  40, -2, -10,  -3,    0, -40,  45,   5, -30,
        11,   9,  -5,   1,   7,  23,   3, -17,  4,   4,   4,    4, -37, -55, -73, -19 } },
    { 1,{ 59,  35,  53, -25,   4,   6,  -2, -14, -2, -10,  -3,    0,  14,  -3,   5,  12,
          11,   9,  -5,   1, -32, -68, -23,  22,-44,   4,  10,   22, -21, -31, -41, -11,
          -5,  23,  -9,  13,  24, -12, -24,  20, -2,  -9,  -3,    5,  10,  35,  40,  50,
         -11,   1, -25, -23,  15,  -1, -13, -41,  4,   4,   4,    4, -64,  17, -10, -10 } },
    }};
};
