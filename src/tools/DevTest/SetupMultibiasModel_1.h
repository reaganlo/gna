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

#include <array>

#include "IModelSetup.h"
#include "DeviceController.h"

class SetupMultibiasModel_1 : public IModelSetup
{
public:
    SetupMultibiasModel_1(DeviceController & deviceCtrl, bool weight2B, bool pwlEn);

    ~SetupMultibiasModel_1();

    void checkReferenceOutput(int modelIndex, int configIndex) const override;

private:
    void sampleAffineLayer();
    void samplePwl(intel_pwl_segment_t *segments, uint32_t nSegments);

    DeviceController & deviceController;

    bool weightsAre2Bytes;
    bool pwlEnabled;

    uint32_t indicesCount;
    uint32_t *indices;

    uint32_t nSegments;

    intel_affine_multibias_func_t multibias_func;
    intel_pwl_func_t pwl;
    intel_affine_multibias_layer_t multibias_layer;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;

    template <class intel_reference_output_type>
    intel_reference_output_type* refOutputAssign(int configIndex) const;

    template <class intel_reference_output_type>
    void compareReferenceValues(unsigned int i, int configIndex) const;

    static const int configMultiBias0 = 0;
    static const int configMultiBias1 = 1;
    static const int configIndexAl_1_1B = 2;
    static const int configIndexAl_1_2B = 3;

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
         -74,  165,   164,   81,
        -123,   84,  -150,  -93,
         -27,  -26,  -154, -231,
         -90, -117,    -7, -101,
         -78,   32,   159, -159,
        -248,  185,   -43,   10,
          66,   18,   227, -147,
          80, -234,   134,   -1,
    };

    const  intel_compound_bias_t compoundBiases[outVecSz * groupingNum] =
    {
        { -74,1,{ 0 } },{ 165,1,{ 0 } },{ 164,1,{ 0 } },{ 81,1,{ 0 } },
        { -123,1,{ 0 } },{ 84,1,{ 0 } },{ -150,1,{ 0 } },{ -93,1,{ 0 } },
        { -27,1,{ 0 } },{ -26,1,{ 0 } },{ -154,1,{ 0 } },{ -231,1,{ 0 } },
        { -90,1,{ 0 } },{ -117,1,{ 0 } },{ -7,1,{ 0 } },{ -101,1,{ 0 } },
        { -78,1,{ 0 } },{ 32,1,{ 0 } },{ 159,1,{ 0 } },{ -159,1,{ 0 } },
        { -248,1,{ 0 } },{ 185,1,{ 0 } },{ -43,1,{ 0 } },{ 10,1,{ 0 } },
        { 66,1,{ 0 } },{ 18,1,{ 0 } },{ 227,1,{ 0 } },{ -147,1,{ 0 } },
        { 80,1,{ 0 } },{ -234,1,{ 0 } },{ 134,1,{ 0 } },{ -1,1,{ 0 } },
    };

    const int32_t ref_output[outVecSz * groupingNum] =
    {
        -101,   -9,  105,  104,
          -1, -270,  -72,  155,
        -389,   45,  -72, -258,
         -58, -166,   52, -135,
        -126, -154, -196, -403,
        -158,  -55,   14, -308,
         -52,   -7, -113, -214,
          20,   56, -103,   10,
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
    };

    static const uint8_t numberOfDnnModels = 4;

    const std::array<int, numberOfDnnModels> refSize =
    {
        sizeof(ref_output) / sizeof(int32_t),
        sizeof(ref_output) / sizeof(int32_t),
        sizeof(ref_output_pwl) / sizeof(int16_t),
        sizeof(ref_output_pwl) / sizeof(int16_t),
    };

    static const int confiMultiBias_1_1B = 0;
    static const int confiMultiBias_1_2B = 1;
    static const int confiMultiBiasPwl_1_1B = 2;
    static const int confiMultiBiasPwl_1_2B = 3;
};
