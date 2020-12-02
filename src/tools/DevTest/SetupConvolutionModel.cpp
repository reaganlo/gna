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

#include "SetupConvolutionModel.h"

#include "ModelUtilities.h"

#include <cstring>
#include <stdexcept>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

SetupConvolutionModel::SetupConvolutionModel(DeviceController & deviceCtrl, bool pwlEn)
    : deviceController{deviceCtrl},
      pwlEnabled{pwlEn}
{

    sampleConvolutionLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = DeviceController::ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);
}

SetupConvolutionModel::~SetupConvolutionModel()
{
    deviceController.Free(memory);

    deviceController.ModelRelease(modelId);
}

template <class intel_reference_output_type>
intel_reference_output_type* SetupConvolutionModel::refOutputAssign(uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(configIndex);
    if (pwlEnabled)
    {
        return (intel_reference_output_type*)ref_outputPwl;
    }
    return (intel_reference_output_type*)ref_output;
}

template <class intel_reference_output_type>
void SetupConvolutionModel::compareReferenceValues(unsigned i, uint32_t configIndex) const
{
    intel_reference_output_type outElemVal = static_cast<const intel_reference_output_type*>(outputBuffer)[i];
    const intel_reference_output_type* refOutput = refOutputAssign<intel_reference_output_type>(configIndex);
    if (refOutput[i] != outElemVal)
    {
        // TODO: how it should notified? return or throw
        throw std::runtime_error("Wrong output");
    }
}

void SetupConvolutionModel::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);

    unsigned int ref_output_size = refSize[configIndex];
    for (unsigned int i = 0; i < ref_output_size; ++i)
    {
        (pwlEnabled) ? compareReferenceValues<int16_t>(i, configIndex) : compareReferenceValues<int32_t>(i, configIndex);
    }
}

void SetupConvolutionModel::samplePwl(Gna2PwlSegment *segments, uint32_t numberOfSegments)
{
    ModelUtilities::GeneratePwlSegments(segments, numberOfSegments);
}

void SetupConvolutionModel::sampleConvolutionLayer()
{
    uint32_t buf_size_filters = Gna2RoundUpTo64(sizeof(filters));
    uint32_t buf_size_inputs = Gna2RoundUpTo64(sizeof(inputs));
    uint32_t buf_size_biases = Gna2RoundUpTo64(sizeof(regularBiases));
    uint32_t buf_size_outputs = Gna2RoundUpTo64(outVecSz * groupingNum * sizeof(int16_t));
    uint32_t buf_size_tmp_outputs = Gna2RoundUpTo64(outVecSz * groupingNum * sizeof(int32_t));

    uint32_t bytes_requested = buf_size_filters + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    if (pwlEnabled)
    {
        uint32_t buf_size_pwl = Gna2RoundUpTo64(nSegments * static_cast<uint32_t>(sizeof(Gna2PwlSegment)));
        bytes_requested += buf_size_pwl;
    }
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

    const auto outputsPerFilter = (inVecSz - nFilterCoefficients)
        / (convStride) + 1;

    operationHolder.InitCnnLegacy(groupingNum, inVecSz, outputsPerFilter, nFilters, nFilterCoefficients, convStride,
        nullptr, nullptr, pinned_filters, pinned_biases);

    if (pwlEnabled)
    {
        void* pinned_pwl = pinned_mem_ptr;

        samplePwl(reinterpret_cast<Gna2PwlSegment*>(pinned_pwl), nSegments);
        operationHolder.AddPwl(nSegments, pinned_pwl, Gna2DataTypeInt16);
    }

    model = { 1, &operationHolder.Get() };
}
