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

#include "SetupDiagonalModel.h"
#include "ModelUtilities.h"

#include "gna-api.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupDiagonalModel::SetupDiagonalModel(DeviceController & deviceCtrl, bool weight2B, bool pwlEn)
    : deviceController{deviceCtrl},
    weightsAre2Bytes{weight2B},
    pwlEnabled{pwlEn}
{
    sampleAffineLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupDiagonalModel::~SetupDiagonalModel()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

template <class intel_reference_output_type>
intel_reference_output_type* SetupDiagonalModel::refOutputAssign(uint32_t configIndex) const
{
    switch (configIndex)
    {
    case configDiagonal_1_1B:
        return (intel_reference_output_type*)ref_output;
    case confiDiagonal_1_2B:
        return (intel_reference_output_type*)ref_output;
    case confiDiagonalPwl_1_1B:
        return (intel_reference_output_type*)ref_output_pwl;
    case confiDiagonalPwl_1_2B:
        return (intel_reference_output_type*)ref_output_pwl;
    default:
        throw std::runtime_error("Invalid configuration index");
    }
}

template <class intel_reference_output_type>
void SetupDiagonalModel::compareReferenceValues(unsigned int i, uint32_t configIndex) const
{
    intel_reference_output_type outElemVal = static_cast<const intel_reference_output_type*>(outputBuffer)[i];
    const intel_reference_output_type* refOutput = refOutputAssign<intel_reference_output_type>(configIndex);
    if (refOutput[i] != outElemVal)
    {
        // TODO: how it should notified? return or throw
        throw std::runtime_error("Wrong output");
    }
}


void SetupDiagonalModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    unsigned int ref_output_size = refSize[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        compareReferenceValues<int16_t>(i, configIndex);
    }
}

void SetupDiagonalModel::sampleAffineLayer()
{
    uint32_t buf_size_weights = static_cast<uint32_t>(weightsAre2Bytes
        ? ALIGN64(sizeof(weights_2B)) : ALIGN64(sizeof(weights_1B)));
    uint32_t buf_size_inputs = ALIGN64(sizeof(inputs));
    uint32_t buf_size_biases = static_cast<uint32_t>(weightsAre2Bytes
        ? ALIGN64(sizeof(regularBiases)) : ALIGN64(sizeof(compoundBiases)));
    uint32_t buf_size_outputs = ALIGN64(outVecSz * groupingNum * sizeof(int32_t));
    uint32_t buf_size_pwl = static_cast<uint32_t>(ALIGN64(nSegments * sizeof(intel_pwl_segment_t)));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs;
    if (pwlEnabled)
    {
        bytes_requested += buf_size_pwl;
    }
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

    operations = static_cast<Gna2Operation*>(calloc(1, sizeof(Gna2Operation)));
    tensors = static_cast<Gna2Tensor*>(calloc(5, sizeof(Gna2Tensor)));

    tensors[0] = Gna2TensorInit2D(inVecSz, groupingNum,
        Gna2DataTypeInt16, nullptr);
    tensors[1] = Gna2TensorInit2D(outVecSz, groupingNum,
        Gna2DataTypeInt32, nullptr);

    tensors[2] = Gna2TensorInit1D(inVecSz,
        weightsAre2Bytes ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        pinned_weights);
    tensors[3] = Gna2TensorInit1D(inVecSz,
        weightsAre2Bytes ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias,
        pinned_biases);
    tensors[4] = Gna2TensorInitDisabled();

    if (pwlEnabled)
    {
        Gna2PwlSegment* pinned_pwl = reinterpret_cast<Gna2PwlSegment*>(pinned_mem_ptr);

        samplePwl(pinned_pwl, nSegments);
        tensors[1] = Gna2TensorInit2D(outVecSz, groupingNum, Gna2DataTypeInt16, nullptr);
        tensors[4] = Gna2TensorInit1D(nSegments, Gna2DataTypePwlSegment, pinned_pwl);
    }

    Gna2OperationInitElementWiseAffine(
        operations, &Allocator,
        &tensors[0], &tensors[1],
        &tensors[2], &tensors[3],
        &tensors[4]
    );

    model = { 1, operations };
}

void SetupDiagonalModel::samplePwl(Gna2PwlSegment* segments, uint32_t numberOfSegments)
{
    ModelUtilities::GeneratePwlSegments(segments, numberOfSegments);
}
