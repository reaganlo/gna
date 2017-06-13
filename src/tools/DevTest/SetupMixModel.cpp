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

#include "gna-api.h"

#include "SetupMixModel.h"
#include "ChainModel.h"

SetupMixModel::SetupMixModel(DeviceController & deviceCtrl)
    : deviceController{deviceCtrl}
{
    ChainModel chainModel; 
    chainModel.Affine(true, true, false).Affine(true, true, false).Multibias(false, true).Convolution(true).Pooling(INTEL_SUM_POOLING).Recurrent(true).Copy().Gmm().Transpose().Transpose();
    uint32_t modelSize = chainModel.GetModelSize();
    uint32_t bytesGranted;

    uint8_t *pinned_memory = deviceController.Alloc(modelSize, &bytesGranted);
    nnet = chainModel.Setup(pinned_memory);

    auto inputBufferSize = chainModel.GetInputBuffersSize();
    auto outputBufferSize = chainModel.GetOutputBuffersSize();

    uint8_t *pinned_free_memory = pinned_memory + modelSize - (inputBufferSize + outputBufferSize);
    inputBuffer = pinned_free_memory;
    pinned_free_memory += inputBufferSize;
    outputBuffer = pinned_free_memory;
    pinned_free_memory += outputBufferSize;

    deviceController.ModelCreate(&nnet, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, GNA_IN, 0, inputBuffer);
    deviceController.BufferAdd(configId, GNA_OUT, nnet.nLayers - 1, outputBuffer);
}

SetupMixModel::~SetupMixModel()
{
    deviceController.ModelRelease(modelId);
    deviceController.Free();
}

void SetupMixModel::checkReferenceOutput() const
{
}
