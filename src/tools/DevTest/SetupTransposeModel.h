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


class SetupTransposeModel : public IModelSetup
{
public:
    SetupTransposeModel(DeviceController & deviceCtrl, uint32_t configIndex);

    ~SetupTransposeModel();

    void checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const override;

private:
    void sampleTransposeLayer(uint32_t configIndex);

    DeviceController & deviceController;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;
    void * memory = nullptr;

    static const int groupingNumConfig1 = 4;
    static const int groupingNumConfig2 = 6;
    static const int outVecSz = 16;

    const int16_t inputs_1[groupingNumConfig1 * inVecSz] =
    {
        -5,  9, -7,  4,  5, -4, -7,  4,  0,  7,  1, -7,  1,  6,  7,  9,
         2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
        -9, -5, -8,  5,  0, -1,  3,  9,  0,  8,  1, -2, -9,  8,  0, -7,
        -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2
    };

    const int16_t inputs_2[groupingNumConfig2 * inVecSz] =
    {
        -80,  236,  230, -237, -251,   96, -126, -121,  -50,  -10, -108,  129, -150,   28,  181,  -68,
         21,   44,  -69,  127,   74, -169,  169,  170,  -24,  147, -133,  197, -145,   43, -244,   76,
        188, -208,  174,  -37,  158,  179,  -86, -252, -197, -125,  -69,  122,  -74, -115,  -62,   94,
        -13,  103,  105,   45,    1,   81,  100, -230,   75,  -84,   85,  115,  -87,  -62, -162, -157,
        212,   49,  -39,  -46, -217,  -70,  -22, -249, -211,   28, -153, -188, -202,  -34,  124,  -13,
        246,  159,   19,  -21, -190,   20,  -19,  116, -205, -125,  -17,  107,   92,   -7,   47,  172,
    };

    const int16_t ref_output_1[outVecSz * groupingNumConfig1] =
    {
        -5,  2, -9, -9,
         9, -4, -5, -8,
        -7,  9, -8, -1,
         4,  8,  5, -4,
         5, -5,  0, -3,
        -4, -1, -1, -7,
        -7,  2,  3, -2,
         4,  9,  9,  3,
         0, -8,  0, -8,
         7, -8,  8,  0,
         1,  8,  1,  1,
        -7,  1, -2,  3,
         1, -7, -9, -4,
         6,  2,  8, -6,
         7, -1,  0, -8,
         9, -1, -7, -2
    };

    const int16_t ref_output_2[outVecSz * groupingNumConfig2] =
    {
         -80,    21,   188,   -13,   212,   246,
         236,    44,  -208,   103,    49,   159,
         230,   -69,   174,   105,   -39,    19,
        -237,   127,   -37,    45,   -46,   -21,
        -251,    74,   158,     1,  -217,  -190,
          96,  -169,   179,    81,   -70,    20,
        -126,   169,   -86,   100,   -22,   -19,
        -121,   170,  -252,  -230,  -249,   116,
         -50,   -24,  -197,    75,  -211,  -205,
         -10,   147,  -125,   -84,    28,  -125,
        -108,  -133,   -69,    85,  -153,   -17,
         129,   197,   122,   115,  -188,   107,
        -150,  -145,   -74,   -87,  -202,    92,
          28,    43,  -115,   -62,   -34,    -7,
         181,  -244,   -62,  -162,   124,    47,
         -68,    76,    94,  -157,  -13,    172,
    };
    static const uint8_t numberOfTransposeModels = 2;
    const std::array<uint32_t, numberOfTransposeModels> refSize
    {{
        sizeof(ref_output_1) / sizeof(int16_t),
        sizeof(ref_output_2) / sizeof(int16_t),
    }};

    const int16_t* refOutputAssign[numberOfTransposeModels] =
    {
        ref_output_1,
        ref_output_2,
    };

    const int16_t* inputs[numberOfTransposeModels] =
    {
        inputs_1,
        inputs_2,
    };

    const uint32_t inputsSize[numberOfTransposeModels] =
    {
        sizeof(inputs_1),
        sizeof(inputs_2),
    };

    const std::array<uint32_t, numberOfTransposeModels> groupingNum
    {{
        groupingNumConfig1,
        groupingNumConfig2
    }};
};
