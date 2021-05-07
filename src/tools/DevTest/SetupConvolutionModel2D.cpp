/*
 INTEL CONFIDENTIAL
 Copyright 2017-2020 Intel Corporation.

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

#include "SetupConvolutionModel2D.h"

#include <cstring>

const int16_t filters[] =
{
    1, 1, 1, 1, 1, 1, 1, 1,

    2, 2, 2, 2, 2, 2, 2, 2,

    3, 3, 3, 3, 3, 3, 3, 3,
};

const int16_t inputs[] = {
    1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
};

const int32_t regularBiases[] = {
    1, 2, 3,
};
//
//const int32_t ref_output[grouping * totalOutputCount] = {
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//   1,1,1,    2,2,2,    3,3,3,    4,4,4,    5,5,5,
//};


const int32_t ref_output[] = {
   1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,
};

SetupConvolutionModel2D::SetupConvolutionModel2D(DeviceController & deviceCtrl)
    : ModelSetup{deviceCtrl, ref_output}
{
    sampleConvolutionLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupConvolutionModel2D::~SetupConvolutionModel2D()
{
   deviceController.Free(memory);

   deviceController.ModelRelease(modelId);
}

void SetupConvolutionModel2D::sampleConvolutionLayer()
{
    constexpr uint32_t filterN = 4;
    uint32_t buf_size_filters = Gna2RoundUpTo64(sizeof(filters));
    uint32_t buf_size_inputs = Gna2RoundUpTo64(sizeof(inputs));
    uint32_t buf_size_biases = Gna2RoundUpTo64(sizeof(regularBiases));
    uint32_t buf_size_outputs = Gna2RoundUpTo64(sizeof(ref_output));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs;
    uint32_t bytes_granted;
    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    void* pinned_filters = pinned_mem_ptr;
    memcpy(pinned_filters, filters, sizeof(filters));
    pinned_mem_ptr += buf_size_filters;

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    pinned_mem_ptr += buf_size_biases;

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;
    outputBuffer = pinned_mem_ptr;

    operationHolder.InitCnn2DPool(16, 1, 1,
        filterN, 8, 1,
        1, 1,
        3, 1,
        Gna2DataTypeInt16, Gna2DataTypeInt32, nullptr, nullptr, pinned_filters, pinned_biases, Gna2PoolingModeMax);

    model = { 1, &operationHolder.Get() };
}
