/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "Device.h"

#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>

#include "FakeDetector.h"
#include "Memory.h"
#include "Validator.h"
#include "HardwareModelSue1.h"

using std::ofstream;
using std::make_shared;
using std::make_unique;
using std::move;

using namespace GNA;

const std::map<const gna_device_kind, const GnaDeviceType> Device::deviceDictionary =
{
    { GNA_SUE, GNA_SUE_CREEK },
    { GNA_SUE_2, GNA_SUE_CREEK_2 },
    { GNA_CNL, GNA_DEV_CNL },
    { GNA_GLK, GNA_DEV_GLK },
    { GNA_ICL, GNA_DEV_ICL },
    { GNA_TGL, GNA_DEV_TGL },
};

void* Device::Dump(gna_model_id modelId, gna_device_kind deviceKind, intel_gna_model_header* modelHeader, intel_gna_status_t* status, intel_gna_alloc_cb customAlloc)
{
    // Validate parameters
    Expect::NotNull(status);
    Expect::NotNull(modelHeader);
    Expect::NotNull(reinterpret_cast<void*>(customAlloc));
    Expect::True(GNA_SUE == deviceKind, GNA_CPUTYPENOTSUPPORTED); // Temporary limitation

    FakeDetector detector{ *ioctlSender, deviceDictionary.at(deviceKind) };

    auto memoryId = 0;
    auto totalMemory = memoryObjects.at(memoryId).get();
    auto& model = totalMemory->GetModel(modelId);
    auto layerCount = model.LayerCount;
    auto gmmCount = model.GetGmmCount();
    auto internalSize = HardwareModelSue1::CalculateDescriptorSize(layerCount, gmmCount);
    size_t const dumpedModelTotalSize = totalMemory->ModelSize + internalSize;

    void * address = customAlloc(dumpedModelTotalSize);

    Expect::NotNull(address);
    memset(address, 0, dumpedModelTotalSize);

    // creating HW layer descriptors directly into dump memory
    auto hwModel = make_unique<HardwareModelSue1>(modelId, model.GetLayers(), *totalMemory, address, *ioctlSender, detector);
    hwModel->Build();

    // copying data..
    void *data = totalMemory->Get() + totalMemory->InternalSize;
    void *dumpData = static_cast<uint8_t*>(address) + internalSize;
    memcpy(dumpData, data, totalMemory->ModelSize);

    /* Determine XNN parameters */
    XNN_LYR * const desc = reinterpret_cast<XNN_LYR*>(address);
    XNN_LYR const * const last_desc = desc + layerCount - 1;
    uint32_t nBytesPerOutput;
    uint32_t outputsOffset;
    if (last_desc->pwl_n_segs == 0
        && ((NN_INTER != last_desc->op)
            && (NN_DEINT != last_desc->op)
            && (NN_COPY != last_desc->op)))
    {
        outputsOffset = (uint8_t*)&last_desc->out_sum_buffer - (uint8_t*)desc;
        nBytesPerOutput = 4;
    }
    else
    {
        outputsOffset = (uint8_t*)&last_desc->out_act_fn_buffer - (uint8_t*)desc;
        nBytesPerOutput = 2;
    }
    uint32_t nOutputs;
    switch (last_desc->op)
    {
    case NN_AFF_AL:
        nOutputs = last_desc->act_list_n_elems * last_desc->n_groups;
        break;
    case NN_CNN:
    {
        uint32_t max_ncoe = (last_desc->n_in_elems - last_desc->cnn_flt_size) / last_desc->cnn_n_flt_stride + 1;
        uint32_t n_outputs_per_filter = (0 == last_desc->flags.pool_param)
            ? max_ncoe
            : (max_ncoe - 1) / last_desc->cnn_pool_stride + 1;
        nOutputs = n_outputs_per_filter * last_desc->cnn_n_flts;
        break;
    }
    default:
        nOutputs = last_desc->n_out_elems * last_desc->n_groups;
        break;
    }

    // filling model header
    uint32_t const gna_mode = 1;
    uint32_t const nBytesPerInput = 2;
    uint32_t nInputs = desc->n_in_elems;
    if (NN_CNN != desc->op)
    {
        nInputs *= desc->n_groups;
    }

    uint32_t inputsOffset = (uint8_t*)&desc->in_buffer - (uint8_t*)desc;
    *modelHeader = { 0, static_cast<uint32_t>(dumpedModelTotalSize), gna_mode, layerCount, nBytesPerInput,
        nBytesPerOutput, nInputs, nOutputs, inputsOffset, outputsOffset };

    return address;
}
