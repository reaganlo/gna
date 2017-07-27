/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include <iostream>
#include <fstream>
#include <memory>

#include "FakeDetector.h"
#include "Memory.h"
#include "Validator.h"

using std::ofstream;
using std::make_shared;
using std::make_unique;
using std::move;

using namespace GNA;

void Device::DumpModel(gna_model_id modelId, gna_device_kind deviceKind, const char * filepath)
{
    auto deviceType = static_cast<GnaDeviceType>(deviceKind);
    FakeDetector detector{ *ioctlSender, deviceType };

    auto memoryId = 0;
    auto totalMemory = memoryObjects.at(memoryId).get();
    auto& model = totalMemory->GetModel(modelId);
    auto layerCount = model.LayerCount;
    auto gmmCount = model.GetGmmCount();
    auto internalSize = CompiledModel::CalculateInternalModelSize(layerCount, gmmCount);

    gna_model_id dumpModelId;
    void * address = totalMemory->Get() + totalMemory->InternalSize - internalSize;

    // using placement new to avoid Memory destructor
    void *memory = malloc(sizeof(Memory));
    auto *dumpMemory = new (memory) Memory{ totalMemory->Id, address, totalMemory->ModelSize, layerCount, gmmCount, *ioctlSender };

    // save original layer descriptors
    void *descriptorsCopy = malloc(totalMemory->InternalSize);
    memcpy(descriptorsCopy, *totalMemory, totalMemory->InternalSize);

    // generating layer descriptors..
    auto hwModel = make_unique<HardwareModel>(modelId, model.GetLayers(), gmmCount, *dumpMemory, *ioctlSender, detector);

    // copying data..
    void *data = totalMemory->Get() + totalMemory->InternalSize;
    void *dumpData = dumpMemory->Get() + dumpMemory->InternalSize;
    memcpy(dumpData, data, totalMemory->ModelSize);

    ofstream dumpStream(filepath, std::ios::out | std::ios::binary);

    // dump SUE model header needed for internal model handling
    if (GNA_SUE_CREEK == static_cast<GnaDeviceType>(deviceKind)
        || GNA_SUE_CREEK_2 == static_cast<GnaDeviceType>(deviceKind))
    {
        auto xnnDescriptor = reinterpret_cast<XNN_LYR*>(dumpMemory->Get());
        uint32_t input_elements = xnnDescriptor->n_in_elems;
        if (NN_CNN != xnnDescriptor->op) input_elements *= xnnDescriptor->n_groups;

        uint32_t inputs_size = input_elements * sizeof(int16_t);
        void * input_buffer = dumpMemory->Get<uint8_t>() + xnnDescriptor->in_buffer;

        /* XNN parameters */
        uint32_t gna_mode = 1;
        uint32_t nInputs = xnnDescriptor->n_in_elems;
        if (NN_CNN != xnnDescriptor->op)
        {
            nInputs *= xnnDescriptor->n_groups;
        }

        uint32_t inputsOffset = offsetof(XNN_LYR, in_buffer);

        XNN_LYR *last_hwLyrDsc = xnnDescriptor + layerCount - 1;
        uint32_t nOutputs;
        uint32_t nBytesPerInput = 2;
        uint32_t nBytesPerOutput;
        uint32_t outputsOffset;
        if (last_hwLyrDsc->pwl_n_segs == 0
            && ((NN_INTER != last_hwLyrDsc->op)
                && (NN_DEINT != last_hwLyrDsc->op)
                && (NN_COPY != last_hwLyrDsc->op)))
        {
            outputsOffset = (uint8_t*)&last_hwLyrDsc->out_sum_buffer - (uint8_t*)xnnDescriptor;
            nBytesPerOutput = 4;
        }
        else
        {
            outputsOffset = (uint8_t*)&last_hwLyrDsc->out_act_fn_buffer - (uint8_t*)xnnDescriptor;
            nBytesPerOutput = 2;
        }

        switch (last_hwLyrDsc->op)
        {
        case NN_AFF_AL:
            nOutputs = last_hwLyrDsc->act_list_n_elems * last_hwLyrDsc->n_groups;
            break;
        case NN_CNN:
        {
            uint32_t max_ncoe = (last_hwLyrDsc->n_in_elems - last_hwLyrDsc->cnn_flt_size) / last_hwLyrDsc->cnn_n_flt_stride + 1;
            uint32_t n_outputs_per_filter = (0 == last_hwLyrDsc->flags.pool_param)
                ? max_ncoe
                : (max_ncoe - 1) / last_hwLyrDsc->cnn_pool_stride + 1;
            nOutputs = n_outputs_per_filter * last_hwLyrDsc->cnn_n_flts;
            break;
        }
        default:
            nOutputs = last_hwLyrDsc->n_out_elems * last_hwLyrDsc->n_groups;
            break;
        }

        uint32_t params[] = { 0, 0, gna_mode, layerCount, nBytesPerInput, nBytesPerOutput, nInputs, nOutputs, inputsOffset, outputsOffset };
        params[1] = sizeof(params) + dumpMemory->GetSize();

        dumpStream.write(reinterpret_cast<const char*>(params), sizeof(params));
    }

    dumpStream.write(dumpMemory->Get<const char>(), dumpMemory->GetSize());
    free(memory);

    // restore layer descriptors
    memcpy(*totalMemory, descriptorsCopy, totalMemory->InternalSize);
    free(descriptorsCopy);
}
