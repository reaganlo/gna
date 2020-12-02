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

#include "SetupGmmModel.h"

#include <cstring>
#include <stdexcept>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupGmmModel::SetupGmmModel(DeviceController & deviceCtrl, bool activeListEn)
    : deviceController{deviceCtrl},
      activeListEnabled{activeListEn}
{
    memset(&model, 0, sizeof(model));
    sampleGmmLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);

    if (activeListEnabled)
    {
        DeviceController::ActiveListAdd(configId, 0, indicesCount, indices);
    }
}

SetupGmmModel::~SetupGmmModel()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

void SetupGmmModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    auto const ref_output_size = refSize[configIndex];
    const auto * const ref_output = refOutputAssign[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        auto const outElemVal = static_cast<const uint32_t*>(outputBuffer)[i];
        if (ref_output[i] != outElemVal)
        {
            // TODO: how it should notified? return or throw
            throw std::runtime_error("Wrong output");
        }
    }
}

void SetupGmmModel::sampleGmmLayer()
{
    uint32_t const batchSize = groupingNum;
    uint32_t const featureVectorLength = inVecSz;

    auto const buf_size_weights = Gna2RoundUpTo64(sizeof(variance)); // note that buffer alignment to 64-bytes is required by GNA HW
    auto const buf_size_inputs = Gna2RoundUpTo64(sizeof(feature_vector));
    auto const buf_size_outputs = Gna2RoundUpTo64(sizeof(ref_output_));
    auto const indicesSize = static_cast<uint32_t>(indicesCount * sizeof(uint32_t));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_outputs;
    if (activeListEnabled)
    {
        bytes_requested += indicesSize;
    }
    uint32_t bytes_granted;

    // call GNAAlloc (obtains pinned memory shared with the device)
    memory = deviceController.Alloc(bytes_requested, &bytes_granted);
    auto * pinned_mem_ptr = static_cast<uint8_t*>(memory);
    memset(pinned_mem_ptr, 0, bytes_granted);

    auto * const pinned_weights = pinned_mem_ptr;
    memcpy(pinned_weights, variance, sizeof(variance));   // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    inputBuffer = pinned_mem_ptr;
    memcpy(inputBuffer, feature_vector, sizeof(feature_vector));      // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;                  // fast-forwards current pinned memory pointer to the next free block

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;                 // fast-forwards the current pinned memory pointer by the space needed for outputs

    if (activeListEnabled)
    {
        indices = (uint32_t*)pinned_mem_ptr;
        memcpy(indices, alIndices, indicesSize);
        pinned_mem_ptr += indicesSize;
    }

    operationHolder.InitGmm(batchSize, featureVectorLength, gmmStates, mixtures, memory, memory, pinned_weights, UINT32_MAX);
    model = { 1, &operationHolder.Get() };
}
