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

#include "SetupMultibiasModel_1.h"

#include "ModelUtilities.h"

#include <cstring>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupMultibiasModel_1::SetupMultibiasModel_1(DeviceController& deviceCtrl, bool weight2B, bool pwlEn)
    : deviceController{ deviceCtrl },
    weightsAre2Bytes{ weight2B },
    pwlEnabled{ pwlEn }
{
    sampleAffineLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupMultibiasModel_1::~SetupMultibiasModel_1()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

template <class intel_reference_output_type>
intel_reference_output_type* SetupMultibiasModel_1::refOutputAssign(uint32_t configIndex) const
{
    switch (configIndex)
    {
    case confiMultiBias_1_1B:
        return (intel_reference_output_type*)ref_output;
    case confiMultiBias_1_2B:
        return (intel_reference_output_type*)ref_output;
    case confiMultiBiasPwl_1_1B:
        return (intel_reference_output_type*)ref_output_pwl;
    case confiMultiBiasPwl_1_2B:
        return (intel_reference_output_type*)ref_output_pwl;
    default:
        throw std::runtime_error("Invalid configuration index");;
    }
}

template <class intel_reference_output_type>
void SetupMultibiasModel_1::compareReferenceValues(unsigned int i, uint32_t configIndex) const
{
    intel_reference_output_type outElemVal = static_cast<const intel_reference_output_type*>(outputBuffer)[i];
    const intel_reference_output_type* refOutput = refOutputAssign<intel_reference_output_type>(configIndex);
    if (refOutput[i] != outElemVal)
    {
        // TODO: how it should notified? return or throw
        throw std::runtime_error("Wrong output");
    }
}

void SetupMultibiasModel_1::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    unsigned int ref_output_size = refSize[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        switch (configIndex)
        {
        case confiMultiBias_1_1B:
            compareReferenceValues<int32_t>(i, configIndex);
            break;
        case confiMultiBias_1_2B:
            compareReferenceValues<int32_t>(i, configIndex);
            break;
        case confiMultiBiasPwl_1_1B:
            compareReferenceValues<int16_t>(i, configIndex);
            break;
        case confiMultiBiasPwl_1_2B:
            compareReferenceValues<int16_t>(i, configIndex);
            break;
        default:
            throw std::runtime_error("Invalid configuration index");
            break;
        }
    }
}

void SetupMultibiasModel_1::sampleAffineLayer()
{
    uint32_t buf_size_weights = weightsAre2Bytes
        ? Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(weights_2B)))
        : Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(weights_1B)));
    uint32_t buf_size_inputs = Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(inputs)));
    uint32_t buf_size_biases = Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(regularBiases)));
    uint32_t buf_size_weight_scales = weightsAre2Bytes
        ? 0 : Gna2RoundUpTo64(static_cast<uint32_t>(sizeof(scaling)));
    uint32_t buf_size_outputs = Gna2RoundUpTo64(
        outVecSz * groupingNum * static_cast<uint32_t>(sizeof(int32_t)));
    uint32_t buf_size_pwl = Gna2RoundUpTo64(
        nSegments * static_cast<uint32_t>(sizeof(Gna2PwlSegment)));

    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases +
        buf_size_weight_scales + buf_size_outputs;

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
    memcpy(pinned_biases, regularBiases, sizeof(regularBiases));

    pinned_mem_ptr += buf_size_biases;

    Gna2WeightScaleFactor* pinned_weight_scales = nullptr;
    if (!weightsAre2Bytes)
    {
        pinned_weight_scales = (Gna2WeightScaleFactor*)pinned_mem_ptr;
        memcpy(pinned_weight_scales, scaling, sizeof(scaling));
        pinned_mem_ptr += buf_size_weight_scales;
    }

    outputBuffer = pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;

    operationHolder.InitAffineEx(inVecSz, outVecSz, groupingNum, nullptr, nullptr,
        pinned_weights, weightsAre2Bytes ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        pinned_biases, Gna2DataTypeInt32);

    if (!weightsAre2Bytes) {
        operationHolder.AddWeightScaleFactor(outVecSz, pinned_weight_scales);
    }

    if (pwlEnabled)
    {
        Gna2PwlSegment* pinned_pwl = reinterpret_cast<Gna2PwlSegment*>(pinned_mem_ptr);

        samplePwl(pinned_pwl, nSegments);
        operationHolder.AddPwl(nSegments, pinned_pwl, Gna2DataTypeInt16);
    }
    operationHolder.AddMBParameters(3, groupingNum);

    model = { 1, &operationHolder.Get() };
}

void SetupMultibiasModel_1::samplePwl(Gna2PwlSegment* segments, uint32_t numberOfSegments)
{
    ModelUtilities::GeneratePwlSegments(segments, numberOfSegments);
}
