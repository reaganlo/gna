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

#include "DeviceController.h"
#include "IModelSetup.h"

#include <array>

class SetupDiagonalModel : public IModelSetup
{
public:
    SetupDiagonalModel(DeviceController & deviceCtrl, bool weight2B, bool pwlEn);

    ~SetupDiagonalModel();

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
    bool pwlEnabled;

    uint32_t nSegments = 64;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;
    void * memory = nullptr;

    static const int outVecSz = 16;

    const int8_t weights_1B[inVecSz] =
    {
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    };

    const int16_t weights_2B[inVecSz] =
    {
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
    };

    const int32_t ref_output[outVecSz * groupingNum] =
    {
         35, -49,  47, -19,
         -6,  12,  18,  -4,
         -2,  -9,  -3,   5,
          4,  -1,  -2,  -4,
        -11,   1, -25, -23,
        -50, -14,  13,  76,
        -44, -44,  52,  10,
        -36,   9,  -6,  -6,
        -13,  -5, -11,  15,
          4,   0,  16,  40,
         -2, -10,  -3,   0,
        -40,  45,   5, -30,
         11,   9,  -5,   1,
          7,  23,   3, -17,
          4,   4,   4,   4,
        -37, -55, -73, -19
    };

    const int16_t ref_output_pwl[outVecSz * groupingNum] =
    {
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
        32735,  32735,  32735,  32735,
    };

    static const uint8_t numberOfDiagonalModels = 4;
    const std::array<unsigned int, numberOfDiagonalModels> refSize {{
        sizeof(ref_output) / sizeof(int32_t),
        sizeof(ref_output) / sizeof(int32_t),
        sizeof(ref_output_pwl) / sizeof(int16_t),
        sizeof(ref_output_pwl) / sizeof(int16_t),
    }};

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

    const int32_t regularBiases[outVecSz] =
    {
        5, 4, -2, 5, -7, -5, 4, -1, 5, 4, -2, 5, -7, -5, 4, -1
    };

    const  Gna2CompoundBias compoundBiases[outVecSz * groupingNum] =
    {
    { 5,1,{ 0 } },
    { 4,1,{ 0 } },
    { -2,1,{ 0 } },
    { 5,1,{ 0 } },
    { -7,1,{ 0 } },
    { -5,1,{ 0 } },
    { 4,1,{ 0 } },
    { -1,1,{ 0 } },
    { 5,1,{ 0 } },
    { 4,1,{ 0 } },
    { -2,1,{ 0 } },
    { 5,1,{ 0 } },
    { -7,1,{ 0 } },
    { -5,1,{ 0 } },
    { 4,1,{ 0 } },
    { -1,1,{ 0 } },
    };

    static const int configDiagonal_1_1B = 0;
    static const int  confiDiagonal_1_2B = 1;
    static const int  confiDiagonalPwl_1_1B = 2;
    static const int  confiDiagonalPwl_1_2B = 3;
};
