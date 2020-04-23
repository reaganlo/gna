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

#include "SetupTransposeModel.h"

#include <cstring>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupTransposeModel::SetupTransposeModel(DeviceController & deviceCtrl, uint32_t configIndex)
    : deviceController{ deviceCtrl }
{
    sampleTransposeLayer(configIndex);

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupTransposeModel::~SetupTransposeModel()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

void SetupTransposeModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    uint32_t ref_output_size = refSize[configIndex];
    const int16_t * ref_output = refOutputAssign[configIndex];
    for (uint32_t i = 0; i < ref_output_size; ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupTransposeModel::sampleTransposeLayer(uint32_t configIndex)
{
    uint32_t buf_size_inputs = Gna2RoundUpTo64(inputsSize[configIndex]);
    uint32_t buf_size_outputs = Gna2RoundUpTo64(
        outVecSz * groupingNum[configIndex] * static_cast<uint32_t>(sizeof(int32_t)));

    uint32_t bytes_requested = buf_size_inputs + buf_size_outputs;
    uint32_t bytes_granted;

    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs[configIndex], inputsSize[configIndex]);
    pinned_mem_ptr += buf_size_inputs;

    outputBuffer = pinned_mem_ptr;

    operationHolder.InitTranspose(groupingNum[configIndex], inVecSz, Gna2DataTypeInt16,
        nullptr, nullptr);
    model = { 1, &operationHolder.Get() };
}
