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

#include "IModelSetup.h"
#include "DeviceController.h"

#include <array>
#include <memory>

class SetupDnnModel_1 : public IModelSetup
{
public:
    SetupDnnModel_1(DeviceController & deviceCtrl, bool weight2B, bool activeListEn, bool pwlEn);

    ~SetupDnnModel_1();

    void checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const override;

private:
    void sampleAffineLayer();
    void samplePwl(Gna2PwlSegment* segments, uint32_t numberOfSegments);

    template <class intel_reference_output_type>
    intel_reference_output_type* refOutputAssign(uint32_t configIndex) const;

    template <class intel_reference_output_type>
    void compareReferenceValues(unsigned int i, uint32_t configIndex) const;

    DeviceController & deviceController;

    bool weightsAre2Bytes;
    bool activeListEnabled;
    bool pwlEnabled;

    uint32_t indicesCount;
    uint32_t *indices;

    uint32_t nSegments = 64;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;
    void * memory = nullptr;

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

    const int16_t inputs[inVecSz * groupingNum] =
    {
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

    const intel_bias_t regularBiases[outVecSz * groupingNum] =
    {
         5, 4, -2, 5,
        -7, -5, 4, -1
    };

    const  intel_compound_bias_t compoundBiases[outVecSz * groupingNum] =
    {
        { 5,1,{ 0 } },{ 4,1,{ 0 } },{ -2,1,{ 0 } },{ 5,1,{ 0 } },
        { -7,1,{ 0 } },{ -5,1,{ 0 } },{ 4,1,{ 0 } },{ -1,1,{ 0 } },
    };

    const int32_t ref_output_model_1[outVecSz * groupingNum] =
    {
        -177,  -85,   29,   28,
          96, -173,   25,  252,
        -160,  274,  157,  -29,
          48,  -60,  158,  -29,
          26,   -2,  -44, -251,
        -173,  -70,   -1, -323,
          99,  144,   38,  -63,
          20,   56, -103,   10
    };
    const int32_t ref_output_modelAl_1[outVecSz * groupingNum / 2] =
    {
        -177, -85,   29,   28,
        -160, 274,  157,  -29,
          26,  -2,  -44, -251,
          20,  56, -103,   10
    };

    const int16_t ref_output_modelPwl_1[outVecSz * groupingNum] =
    {
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32736,
        32735,  32736,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735
    };

    const int16_t ref_output_modelAlPwl_1[outVecSz / 2 * groupingNum] =
    {
        32735,  32735,  32735,  32735,
        32735,  32736,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
    };

    const uint32_t alIndices[outVecSz / 2]
    {
        0, 2, 4, 7
    };

    static const uint8_t numberOfDnnModels = 8;

    const std::array<uint32_t, numberOfDnnModels> refSize
    {{
        sizeof(ref_output_model_1) / sizeof(int32_t),
        sizeof(ref_output_model_1) / sizeof(int32_t),
        sizeof(ref_output_modelAl_1) / sizeof(int32_t),
        sizeof(ref_output_modelAl_1) / sizeof(int32_t),
        sizeof(ref_output_modelPwl_1) / sizeof(int16_t),
        sizeof(ref_output_modelPwl_1) / sizeof(int16_t),
        sizeof(ref_output_modelAlPwl_1) / sizeof(int16_t),
        sizeof(ref_output_modelAlPwl_1) / sizeof(int16_t),
    }};

    static const int configDnn1_1B = 0;
    static const int configDnn1_2B = 1;
    static const int configDnnAl_1_1B = 2;
    static const int configDnnAl_1_2B = 3;
    static const int configDnnPwl_1_1B = 4;
    static const int configDnnPwl_1_2B = 5;
    static const int configDnnAlPwl_1_1B = 6;
    static const int configDnnAlPwl_1_2B = 7;
};
