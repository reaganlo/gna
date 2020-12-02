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

#include "SetupPoolingModel.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupPoolingModel::SetupPoolingModel(DeviceController & deviceCtrl)
    : deviceController{deviceCtrl}
{
    samplePoolingLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupPoolingModel::~SetupPoolingModel()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

void SetupPoolingModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    UNREFERENCED_PARAMETER(configIndex);
    for (unsigned int i = 0; i < sizeof(ref_output) / sizeof(int16_t); ++i)
    {
        int16_t outElemVal = static_cast<const int16_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupPoolingModel::samplePwl(Gna2PwlSegment *segments, uint32_t numberOfSegments)
{
    auto xBase = -600;
    auto xBaseInc = 2u * static_cast<uint32_t>(abs(xBase)) / numberOfSegments;
    auto yBase = xBase;
    auto yBaseInc = 1;
    for (auto i = uint32_t{0}; i < numberOfSegments; i++, xBase += xBaseInc, yBase += yBaseInc, yBaseInc++)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = static_cast<int16_t>(yBase);
        segments[i].Slope = 1;
    }
}

void SetupPoolingModel::samplePoolingLayer()
{
    uint32_t buf_size_filters = Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(filters)));
    uint32_t buf_size_inputs = Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(inputs)));
    uint32_t buf_size_biases = Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(regularBiases)));
    uint32_t buf_size_outputs = Gna2RoundUpTo64(
            outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int16_t)));
    uint32_t buf_size_tmp_outputs = Gna2RoundUpTo64(
            outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int32_t)));
    uint32_t buf_size_pwl = Gna2RoundUpTo64(
            nSegments * static_cast<uint32_t>(sizeof(Gna2PwlSegment)));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs + buf_size_pwl;
    uint32_t bytes_granted;

    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    uint8_t* pinned_mem_ptr = static_cast<uint8_t*>(memory);

    void* pinned_filters = pinned_mem_ptr;
    memcpy(pinned_filters, filters, sizeof(filters));
    pinned_mem_ptr += buf_size_filters;

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    pinned_mem_ptr += buf_size_inputs;

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    pinned_mem_ptr += buf_size_biases;

    pinned_mem_ptr += buf_size_tmp_outputs;

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    void* pinned_pwl = pinned_mem_ptr;

    const uint32_t cnnStride = 48;
    operationHolder.InitCnnLegacy(groupingNum, inVecSz, 1, nFilters,
        nFilterCoefficients, cnnStride, nullptr, nullptr,
        pinned_filters, pinned_biases);
    operationHolder.AddPooling(Gna2PoolingModeSum, 6, 6);

    operationHolder.AddPwl(nSegments, pinned_pwl);
    samplePwl(static_cast<Gna2PwlSegment*>(pinned_pwl), nSegments);
    model = { 1, &operationHolder.Get() };
}

