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

#include "SetupDnnModel_Multibuffer.h"

#include "ModelUtilities.h"

#include <cstring>

#define UNREFERENCED_PARAMETER(P) ((void)(P))

typedef uint8_t     __MultibufferB_RES;       // 1B of reserved memory

SetupDnnModel_Multibuffer::SetupDnnModel_Multibuffer(DeviceController & deviceCtrl, bool weight2B, bool activeListEn, bool pwlEn)
    : deviceController{ deviceCtrl },
    weightsAre2Bytes{ weight2B },
    activeListEnabled{ activeListEn },
    pwlEnabled{ pwlEn }
{
    sampleAffineLayer();

    deviceController.ModelCreate(&model, &modelId);

    configId = deviceController.ConfigAdd(modelId);

    DeviceController::BufferAddIO(configId, 0, inputBuffer, outputBuffer);

    if (activeListEnabled)
    {
        deviceController.ActiveListAdd(configId, 0, indicesCount, indices);
    }

#if HW_VERBOSE == 1
    constexpr uint32_t nActions = 17;

    dbg_action afterActions[nActions];
    afterActions[0].action_type = GnaSleep;
    afterActions[0].timeout = 100;

    afterActions[1].action_type = GnaLogMessage;
    afterActions[1].log_message = "Log message test\n";

    afterActions[2].action_type = GnaZeroMemory;
    afterActions[2].outputs = outputBuffer;
    afterActions[2].outputs_size = outVecSz * groupingNum * sizeof(int32_t);

    afterActions[3].action_type = GnaReadRegister;
    afterActions[3].gna_register = GNA_STS;

    afterActions[4].action_type = GnaDumpMmio;

    afterActions[5].action_type = GnaDumpPageDirectory;

    afterActions[6].action_type = GnaDumpMemory;
    afterActions[6].filename = "dump.bin";

    afterActions[7].action_type = GnaDumpXnnDescriptor;
    afterActions[7].layer_number = 0;

    afterActions[8].action_type = GnaSetXnnDescriptor;
    afterActions[8].layer_number = 0;
    afterActions[8].xnn_offset = 0;
    afterActions[8].xnn_value = 0x13; // invalid op
    afterActions[8].xnn_value_size = GNA_SET_BYTE;

    afterActions[9] = afterActions[7];

    // start accelerator
    afterActions[10].action_type = GnaWriteRegister;
    afterActions[10].gna_register = GNA_CTRL;
    afterActions[10].reg_operation = Or;
    afterActions[10].reg_value = 1;

    afterActions[11].action_type = GnaReadRegister;
    afterActions[11].gna_register = GNA_STS;

    afterActions[12].action_type = GnaSleep;
    afterActions[12].timeout = 200;

    afterActions[13].action_type = GnaReadRegister;
    afterActions[13].gna_register = GNA_STS;

    // abort
    afterActions[14].action_type = GnaWriteRegister;
    afterActions[14].gna_register = GNA_CTRL;
    afterActions[14].reg_operation = Or;
    afterActions[14].reg_value = 4;

    afterActions[15] = afterActions[12];

    afterActions[16].action_type = GnaReadRegister;
    afterActions[16].gna_register = GNA_STS;

    deviceController.AfterscoreDebug(modelId, nActions, afterActions);
#endif

    //deviceController.DumpModel(modelId, "dump.bin");
}

SetupDnnModel_Multibuffer::~SetupDnnModel_Multibuffer()
{
    deviceController.Free(baseMemory);
    deviceController.Free(ioMemory);
    if (activeListEnabled)
    {
        deviceController.Free(alMemory);
    }
    if (pwlEnabled)
    {
        deviceController.Free(pwlMemory);
    }

    deviceController.ModelRelease(modelId);
}

template <class intel_reference_output_type>
intel_reference_output_type* SetupDnnModel_Multibuffer::refOutputAssign(uint32_t configIndex) const
{
    switch (configIndex)
    {
    case configDnn1_1B:
        return (intel_reference_output_type*)ref_output_model_1;
    case configDnn1_2B:
        return (intel_reference_output_type*)ref_output_model_1;
    case configDnnAl_1B:
        return (intel_reference_output_type*)ref_output_modelAl_1;
    case configDnnAl_2B:
        return (intel_reference_output_type*)ref_output_modelAl_1;
    case configDnnPwl_1B:
        return (intel_reference_output_type*)ref_output_modelPwl_1;
    case configDnnPwl_2B:
        return (intel_reference_output_type*)ref_output_modelPwl_1;
    case configDnnAlPwl_1B:
        return (intel_reference_output_type*)ref_output_modelAlPwl_1;
    case configDnnAlPwl_2B:
        return (intel_reference_output_type*)ref_output_modelAlPwl_1;
    default:
        throw std::runtime_error("Invalid configuration index");;
    }
}

template <class intel_reference_output_type>
void SetupDnnModel_Multibuffer::compareReferenceValues(uint32_t i, uint32_t configIndex) const
{
    intel_reference_output_type outElemVal = static_cast<const intel_reference_output_type*>(outputBuffer)[i];
    const intel_reference_output_type* refOutput = refOutputAssign<intel_reference_output_type>(configIndex);
    if (refOutput[i] != outElemVal)
    {
        // TODO: how it should notified? return or throw
        throw std::runtime_error("Wrong output");
    }
}

void SetupDnnModel_Multibuffer::checkReferenceOutput(uint32_t modelIndex, uint32_t configIndex) const
{
    UNREFERENCED_PARAMETER(modelIndex);
    uint32_t ref_output_size = refSize[configIndex];
    for (uint32_t i = 0; i < ref_output_size; ++i)
    {
        switch (configIndex)
        {
        case configDnn1_1B:
            compareReferenceValues<int32_t>(i, configIndex);
            break;
        case configDnn1_2B:
            compareReferenceValues<int32_t>(i, configIndex);
            break;
        case configDnnAl_1B:
            compareReferenceValues<int32_t>(i, configIndex);
            break;
        case configDnnAl_2B:
            compareReferenceValues<int32_t>(i, configIndex);
            break;
        case configDnnPwl_1B:
            compareReferenceValues<int16_t>(i, configIndex);
            break;
        case configDnnPwl_2B:
            compareReferenceValues<int16_t>(i, configIndex);
            break;
        case configDnnAlPwl_1B:
            compareReferenceValues<int16_t>(i, configIndex);
            break;
        case configDnnAlPwl_2B:
            compareReferenceValues<int16_t>(i, configIndex);
            break;
        default:
            throw std::runtime_error("Invalid configuration index");
            break;
        }
    }
}

void SetupDnnModel_Multibuffer::sampleAffineLayer()
{
    uint32_t buf_size_weights = static_cast<uint32_t>(weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(sizeof(weights_2B)) : ModelUtilities::CastAndRoundUpTo64(sizeof(weights_1B)));
    uint32_t buf_size_inputs = ModelUtilities::CastAndRoundUpTo64(sizeof(inputs));
    uint32_t buf_size_biases = weightsAre2Bytes
        ? ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(regularBiases)))
        : ModelUtilities::CastAndRoundUpTo64(static_cast<uint32_t>(sizeof(compoundBiases)));
    uint32_t buf_size_outputs = static_cast<uint32_t>(ModelUtilities::CastAndRoundUpTo64(outVecSz * groupingNum * sizeof(int32_t)));
    uint32_t buf_size_tmp_outputs = static_cast<uint32_t>(ModelUtilities::CastAndRoundUpTo64(outVecSz * groupingNum * sizeof(int32_t)));
    uint32_t buf_size_pwl = static_cast<uint32_t>(ModelUtilities::CastAndRoundUpTo64(nSegments * sizeof(Gna2PwlSegment)));

    uint32_t bytes_requested_al = static_cast<uint32_t>(ModelUtilities::CastAndRoundUpTo64(indicesCount * sizeof(uint32_t)));
    uint32_t bytes_requested_base = buf_size_weights + buf_size_biases;
    uint32_t bytes_requested_io = buf_size_inputs + buf_size_outputs;
    uint32_t bytes_requested_pwl = buf_size_pwl + buf_size_tmp_outputs;

    uint32_t bytes_granted;

    ioMemory = deviceController.Alloc(bytes_requested_io, &bytes_granted);
    uint8_t* ioMemoryPosition = static_cast<uint8_t*>(ioMemory);

    inputBuffer = ioMemoryPosition;
    memcpy(inputBuffer, inputs, sizeof(inputs));
    ioMemoryPosition += buf_size_inputs;

    outputBuffer = ioMemoryPosition;

    baseMemory = deviceController.Alloc(bytes_requested_base, &bytes_granted);
    uint8_t* baseMemoryPosition = static_cast<uint8_t*>(baseMemory);

    void* pinned_weights = baseMemoryPosition;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_weights, weights_2B, sizeof(weights_2B));
    }
    else
    {
        memcpy(pinned_weights, weights_1B, sizeof(weights_1B));
    }
    baseMemoryPosition += buf_size_weights;

    int32_t* pinned_biases = (int32_t*)baseMemoryPosition;
    if (weightsAre2Bytes)
    {
        memcpy(pinned_biases, regularBiases, sizeof(regularBiases));
    }
    else
    {
        memcpy(pinned_biases, compoundBiases, sizeof(compoundBiases));
    }

    if (activeListEnabled)
    {
        alMemory = deviceController.Alloc(bytes_requested_al, &bytes_granted);

        indices = (uint32_t*)alMemory;
        memcpy(indices, alIndices, indicesCount * sizeof(uint32_t));
    }

    operationHolder.InitAffineEx(inVecSz, outVecSz, groupingNum, inputBuffer, outputBuffer,
        pinned_weights, weightsAre2Bytes ? Gna2DataTypeInt16 : Gna2DataTypeInt8,
        pinned_biases, weightsAre2Bytes ? Gna2DataTypeInt32 : Gna2DataTypeCompoundBias);

    if (pwlEnabled)
    {
        pwlMemory = deviceController.Alloc(bytes_requested_pwl, &bytes_granted);

        Gna2PwlSegment* pinned_pwl = reinterpret_cast<Gna2PwlSegment*>(pwlMemory);

        samplePwl(pinned_pwl, nSegments);
        operationHolder.AddPwl(nSegments, pinned_pwl, Gna2DataTypeInt16);
    }

    model = { 1, &operationHolder.Get() };
}

void SetupDnnModel_Multibuffer::samplePwl(Gna2PwlSegment* segments, uint32_t numberOfSegments)
{
    ModelUtilities::GeneratePwlSegments(segments, numberOfSegments);
}

