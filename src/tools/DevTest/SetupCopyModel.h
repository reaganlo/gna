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

class SetupCopyModel : public IModelSetup
{
public:
    SetupCopyModel(DeviceController & deviceCtrl, uint32_t nCopyColumns, uint32_t nCopyRows);

    ~SetupCopyModel();

    void checkReferenceOutput(int modelIndex, int configIndex) const override;

private:
    void sampleCopyLayer(uint32_t nCopyColumns, uint32_t nCopyRows);

    DeviceController & deviceController;

    intel_copy_layer_t copy_layer;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;

    static const int outVecSz = 16;

    const int16_t inputs[groupingNum * inVecSz] =
    {
        -5,  9, -7,  4,  5, -4, -7,  4,  0,  7,  1, -7,  1,  6,  7,  9,
         2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
        -9, -5, -8,  5,  0, -1,  3,  9,  0,  8,  1, -2, -9,  8,  0, -7,
        -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2
    };

    const int16_t ref_output_model_1[groupingNum * outVecSz] =
    {
        -5,  9, -7,  4,  5, -4, -7,  4,  0,  7,  1, -7,  1,  6,  7,  9,
        2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
        -9, -5, -8,  5,  0, -1,  3,  9,  0,  8,  1, -2, -9,  8,  0, -7,
        -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2
    };

    const int16_t ref_output_model_2[groupingNum * outVecSz / 2] =
    {
        -5,  9, -7,  4,  5, -4, -7,  4,  0,  7,  1, -7,  1,  6,  7,  9,
        2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
    };

    const int16_t ref_output_model_3[groupingNum * outVecSz] =
    {
        -5,  9, -7,  4,  5, -4, -7,  4, 0, 0, 0, 0, 0, 0, 0, 0,
        2, -4,  9,  8, -5, -1,  2,  9, 0, 0, 0, 0, 0, 0, 0, 0,
        -9, -5, -8,  5,  0, -1,  3,  9, 0, 0, 0, 0, 0, 0, 0, 0,
        -9, -8, -1, -4, -3, -7, -2,  3, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    const int16_t ref_output_model_4[groupingNum * outVecSz / 2] =
    {
        -5,  9, -7,  4,  5, -4, -7,  4, 0, 0, 0, 0, 0, 0, 0, 0,
        2, -4,  9,  8, -5, -1,  2,  9, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    static const uint8_t numberOfCopyModels = 4;

    const std::array<unsigned int, numberOfCopyModels> refSize =
    {
        sizeof(ref_output_model_1) / sizeof(int16_t),
        sizeof(ref_output_model_2) / sizeof(int16_t),
        sizeof(ref_output_model_3) / sizeof(int16_t),
        sizeof(ref_output_model_4) / sizeof(int16_t),
    };

    const int16_t* refOutputAssign[numberOfCopyModels] =
    {
        ref_output_model_1,
        ref_output_model_2,
        ref_output_model_3,
        ref_output_model_4,
    };
};
