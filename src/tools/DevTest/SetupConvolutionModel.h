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

class SetupConvolutionModel : public IModelSetup
{
public:
    SetupConvolutionModel(DeviceController & deviceCtrl, bool pwlEn);

    ~SetupConvolutionModel();

    void checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const override;

private:
    void sampleConvolutionLayer();
    void samplePwl(intel_pwl_segment_t *segments, uint32_t numberOfSegments);

    DeviceController & deviceController;

    bool pwlEnabled;
    uint32_t nSegments = 64;

    intel_pwl_func_t pwl;
    intel_convolutional_layer_t convolution_layer;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;
    void * memory = nullptr;

    template<class intel_reference_output_type>
    intel_reference_output_type* refOutputAssign(uint32_t configIndex) const;

    template<class intel_reference_output_type>
    void compareReferenceValues(unsigned i, uint32_t configIndex) const;

    static const int groupingNum = 1;
    static const int nFilters = 4;
    static const int nFilterCoefficients = 48;
    static const int inVecSz = 96;

    const int16_t filters[nFilters * nFilterCoefficients] =
    {
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,
        -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
         2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,

         0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
        -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
        -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,

         0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
         2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7,
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,

        -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,
         2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
         0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
    };

    const int16_t inputs[inVecSz * groupingNum] =
    {
        -5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
         2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1,
        -9, -5, -8,  5, 0, -1,  3,  9, 0,  8,  1, -2, -9,  8,  0, -7,

        -9, -8, -1, -4, -3, -7, -2,  3, -8,  0,  1,  3, -4, -6, -8, -2,
        -5,  9, -7,  4, 5, -4, -7,  4, 0,  7,  1, -7, 1,  6,  7,  9,
         2, -4,  9,  8, -5, -1,  2,  9, -8, -8,  8,  1, -7,  2, -1, -1
    };

    const intel_bias_t regularBiases[outVecSz * groupingNum] =
    {
        5, 4, -2, 5
    };

    const int32_t ref_output[outVecSz * groupingNum] =
    {
        -83, -108, -127, 666, 558, -152, 131, -171
    };

    const int16_t ref_outputPwl[outVecSz * groupingNum] =
    {
        32735, 32735, 32735, 32737, 32737, 32735, 32735, 32735
    };
    static const uint8_t numberOfConvolutionModels = 2;
    const std::array<unsigned int, numberOfConvolutionModels> refSize
    {{
        sizeof(ref_output) / sizeof(int32_t),
        sizeof(ref_outputPwl) / sizeof(int16_t),
    }};
};
