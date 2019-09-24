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

#include <cstring>
#include <cstdlib>
#include <iostream>

#include "ModelSetup.h"

#define UNREFERENCED_PARAMETER(P) ((void)(P))

ModelSetup::ModelSetup(DeviceController & deviceCtrl, intel_nnet_type_t nnetModel,
    const void* referenceOutputIn) :
    deviceController{deviceCtrl},
    nnet{nnetModel},
    referenceOutput{referenceOutputIn}
{
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));
}

ModelSetup::~ModelSetup()
{
    free(nnet.pLayers);
}

void ModelSetup::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    UNREFERENCED_PARAMETER(configIndex);

    auto& outputLayer = nnet.pLayers[nnet.nLayers - 1];
    auto outputSize = outputLayer.nOutputRows * outputLayer.nOutputColumns;
    for (unsigned i = 0; i < outputSize; ++i)
    {
        switch (outputLayer.nBytesPerOutput)
        {
        case sizeof (int8_t):
            if (static_cast<const int8_t*>(referenceOutput)[i] != static_cast<const int8_t*>(outputBuffer)[i])
            {
                throw std::runtime_error("Wrong output");
            }
        break;
        case sizeof (int16_t):
            if (static_cast<const int16_t*>(referenceOutput)[i] != static_cast<const int16_t*>(outputBuffer)[i])
            {
                throw std::runtime_error("Wrong output");
            }
        break;
        case sizeof (int32_t):
            if (static_cast<const int32_t*>(referenceOutput)[i] != static_cast<const int32_t*>(outputBuffer)[i])
            {
                throw std::runtime_error("Wrong output");
            }
        break;
        case sizeof (int64_t):
            if (static_cast<const int64_t*>(referenceOutput)[i] != static_cast<const int64_t*>(outputBuffer)[i])
            {
                throw std::runtime_error("Wrong output");
            }
        break;
        default:
            throw std::runtime_error("Invalid output data mode");
        break;
        }
    }
}
