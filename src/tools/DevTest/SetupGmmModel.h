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

class SetupGmmModel : public IModelSetup
{
public:
    SetupGmmModel(DeviceController & deviceCtrl, bool activeListEn);

    ~SetupGmmModel();

    void checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const override;

private:
    void sampleGmmLayer(intel_nnet_type_t& hNnet);

    DeviceController & deviceController;

    bool activeListEnabled;
    uint32_t indicesCount;
    uint32_t* indices;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;
    void * memory = nullptr;

    static const int groupingNum = 1;
    static const int inVecSz = 32;
    static const int inVecSzRow = 2;
    static const int outVecSzAl = 4;


    int16_t variance[outVecSzAl * inVecSz] =
    {                                          // sample weight matrix (8 rows, 16 cols)
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
    };

    uint8_t feature_vector[inVecSzRow * inVecSz] =
    {
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,

    };

    int32_t Gconst[inVecSz] =
    {      // sample bias vector, will get added to each of the four output vectors
        1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,
    };

    const uint32_t alIndices[outVecSzAl]
    {
        0, 2, 4, 7
    };

    const int32_t ref_output_[outVecSz * groupingNum] =
    {
        1, 1, 1, 1, 1, 1, 1, 1,
    };

    const int32_t ref_output_Al[outVecSzAl * groupingNum] =
    {
        1, 1, 1, 1
    };

    static const uint8_t numberOfGmmModels = 2;

    std::array<unsigned int, numberOfGmmModels> refSize
    {{
        sizeof(ref_output_) / sizeof(int32_t),
        sizeof(ref_output_Al) / sizeof(int32_t)
    }};

    const int32_t* refOutputAssign[numberOfGmmModels] =
    {
        ref_output_,
        ref_output_Al,
    };
};
