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

#include "SetupRecurrentModel.h"

#include "gna-api.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupRecurrentModel::SetupRecurrentModel(DeviceController & deviceCtrl, bool wght2B)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{wght2B}
{
    sampleRnnLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    deviceController.BufferAdd(configId, InputComponent, 0, inputBuffer);
    deviceController.BufferAdd(configId, OutputComponent, 0, outputBuffer);
}

SetupRecurrentModel::~SetupRecurrentModel()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

void SetupRecurrentModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    UNREFERENCED_PARAMETER(configIndex);
    for (unsigned int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupRecurrentModel::samplePwl(Gna2PwlSegment* segments, uint32_t numberOfSegments)
{
    auto xBase = -200;
    auto xBaseInc = 2u * static_cast<uint32_t>(abs(xBase)) / numberOfSegments;
    auto yBase = -200;
    auto yBaseInc = 1;
    for (auto i = uint32_t{ 0 }; i < numberOfSegments; i++, xBase += xBaseInc, yBase += yBaseInc, yBaseInc++)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = static_cast<int16_t>(yBase);
        segments[i].Slope = 1;
    }
}

void SetupRecurrentModel::sampleRnnLayer()
{
    parameters = calloc(1, sizeof(uint32_t));
    auto const delay = static_cast<uint32_t*>(parameters);
    *delay = 3;

    uint32_t buf_size_weights = weightsAre2Bytes
        ? ALIGN64(static_cast<uint32_t>(sizeof(weights_2B)))
        : ALIGN64(static_cast<uint32_t>(sizeof(weights_1B)));
    uint32_t buf_size_inputs = ALIGN64(static_cast<uint32_t>(sizeof(inputs)));
    uint32_t buf_size_biases = weightsAre2Bytes
        ? ALIGN64(static_cast<uint32_t>(sizeof(regularBiases)))
        : ALIGN64(static_cast<uint32_t>(sizeof(compoundBiases)));
    uint32_t buf_size_outputs = ALIGN64(
        outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int16_t)));
    uint32_t buf_size_tmp_outputs = ALIGN64(
        (*delay + 1) * outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int32_t)));
    uint32_t buf_size_pwl = ALIGN64(
        nSegments * static_cast<uint32_t>(sizeof(Gna2PwlSegment)));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs + buf_size_pwl;
    uint32_t bytes_granted;

    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    void* pinned_weights = pinned_mem_ptr;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_weights, weights_2B, sizeof(weights_2B));
    }
    else
    {
        memcpy(pinned_weights, weights_1B, sizeof(weights_1B));
    }
    pinned_mem_ptr += buf_size_weights;

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

    int32_t* pinned_biases = (int32_t*)pinned_mem_ptr;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    }
    else
    {
        memcpy(pinned_biases, compoundBiases, sizeof(compoundBiases));
    }
    pinned_mem_ptr += buf_size_biases;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    pwl.nSegments = nSegments;
    pwl.pSegments = reinterpret_cast<struct Gna2PwlSegment*>(pinned_mem_ptr);
    samplePwl(pwl.pSegments, pwl.nSegments);

    operations = static_cast<Gna2Operation*>(calloc(1, sizeof(Gna2Operation)));
    tensors = static_cast<Gna2Tensor*>(calloc(5, sizeof(Gna2Tensor)));

    tensors[0] = Gna2TensorInit2D(groupingNum, 8,
        Gna2DataTypeInt16, nullptr);
    tensors[1] = Gna2TensorInit2D(groupingNum, outVecSz,
        Gna2DataTypeInt16, nullptr);

    tensors[2] = Gna2TensorInit2D(32, 40,
        weightsAre2Bytes ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        pinned_weights);

    tensors[3] = Gna2TensorInit1D(outVecSz * 4,
        weightsAre2Bytes ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias,
        pinned_biases);

    tensors[4] = Gna2TensorInitActivation(nSegments, pwl.pSegments);


    Gna2OperationInitRecurrent(
        operations, &Allocator,
        &tensors[0], &tensors[1],
        &tensors[2], &tensors[3],
        &tensors[4], delay);

    model = { 1, operations };
}
