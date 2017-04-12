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
    auto& model = modelContainer.GetModel(modelId);

    auto deviceType = static_cast<GnaDeviceType>(deviceKind);

    FakeDetector detector{ deviceType };
    model.CompileHardwareModel(detector);

    auto layerCount = model.LayerCount;
    auto internalSize = ModelCompiler::CalculateInternalModelSize(layerCount, model.GetGmmCount());
    auto modelSize = totalMemory->GetSize() - ModelCompiler::MaximumInternalModelSize;

    auto hwLyrDsc = totalMemory->Get<XNN_LYR>();
    auto modelBuffers = totalMemory->Get() + ModelCompiler::MaximumInternalModelSize;

    auto dumpMemory = make_unique<Memory>(internalSize + modelSize);
    auto dumpModel = make_unique<CompiledModel>(model.Id, model.UserModel, *dumpMemory);
    dumpModel->CompileSoftwareModel();
    dumpModel->CompileHardwareModel(detector);

    auto userBuffer = dumpMemory->Get();
    auto dumpDsc = reinterpret_cast<XNN_LYR*>(userBuffer);

    // TODO: after having model built from XML, this should go away
    uint32_t patch_offset = ModelCompiler::MaximumInternalModelSize - internalSize;
    for (int i = 0; i < layerCount; i++)
    {
        if (dumpDsc[i].act_list_buffer != 0)
            dumpDsc[i].act_list_buffer = hwLyrDsc[i].act_list_buffer - patch_offset;

        if (dumpDsc[i].aff_const_buffer != 0)
            dumpDsc[i].aff_const_buffer = hwLyrDsc[i].aff_const_buffer - patch_offset;

        if (dumpDsc[i].aff_weight_buffer != 0)
            dumpDsc[i].aff_weight_buffer = hwLyrDsc[i].aff_weight_buffer - patch_offset;

        // aff_weight_buffer and cnn_flt_buffer are the same through union
        // if(hwLyrDsc[i].cnn_flt_buffer != 0) hwLyrDsc[i].cnn_flt_buffer = hwLyrDsc[i].cnn_flt_buffer - patch_offset;

        if (dumpDsc[i].in_buffer != 0)
            dumpDsc[i].in_buffer = hwLyrDsc[i].in_buffer - patch_offset;

        if (dumpDsc[i].out_act_fn_buffer != 0)
            dumpDsc[i].out_act_fn_buffer = hwLyrDsc[i].out_act_fn_buffer - patch_offset;

        if (dumpDsc[i].out_sum_buffer != 0)
            dumpDsc[i].out_sum_buffer = hwLyrDsc[i].out_sum_buffer - patch_offset;

        if (dumpDsc[i].pwl_seg_def_buffer != 0)
            dumpDsc[i].pwl_seg_def_buffer = hwLyrDsc[i].pwl_seg_def_buffer - patch_offset;

        if (dumpDsc[i].rnn_out_fb_buffer != 0)
            dumpDsc[i].rnn_out_fb_buffer = hwLyrDsc[i].rnn_out_fb_buffer - patch_offset;
    }

    ofstream dumpStream(filepath, std::ios::out | std::ios::binary);

    // dump SUE model header needed for internal model handling
    // TBD: considering universal model header for all platforms
    if (GNA_SUE_CREEK == deviceKind || GNA_SUE_CREEK_2 == deviceKind)
    {
        uint32_t input_elements = dumpDsc->n_in_elems;
        if (NN_CNN != dumpDsc->op) input_elements *= dumpDsc->n_groups;

        uint32_t inputs_size = input_elements * sizeof(int16_t);
        void * input_buffer = (uint8_t*)userBuffer + dumpDsc->in_buffer;

        /* XNN parameters */
        uint32_t gna_mode = 1;
        uint32_t nInputs = dumpDsc->n_in_elems;
        if (NN_CNN != dumpDsc->op)
        {
            nInputs *= dumpDsc->n_groups;
        }


        uint32_t inputsOffset = dumpModel->GetHardwareOffset(&dumpDsc->in_buffer);

        XNN_LYR *last_hwLyrDsc = dumpDsc + layerCount - 1;
        uint32_t nOutputs;
        uint32_t nBytesPerInput = 2;
        uint32_t nBytesPerOutput;
        uint32_t outputsOffset;
        if (last_hwLyrDsc->pwl_n_segs == 0
            && ((NN_INTER != last_hwLyrDsc->op)
                && (NN_DEINT != last_hwLyrDsc->op)
                && (NN_COPY != last_hwLyrDsc->op)))
        {
            outputsOffset = dumpModel->GetHardwareOffset(&last_hwLyrDsc->out_sum_buffer);
            nBytesPerOutput = 4;
        }
        else
        {
            outputsOffset = dumpModel->GetHardwareOffset(&last_hwLyrDsc->out_act_fn_buffer);
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

    dumpStream.write(reinterpret_cast<const char*>(dumpMemory->Get()), internalSize);
    dumpStream.write(reinterpret_cast<const char*>(modelBuffers), modelSize);
}