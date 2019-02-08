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
#include "Expect.h"
#include "HardwareModelSue1.h"

using std::ofstream;
using std::make_shared;
using std::make_unique;
using std::move;

using namespace GNA;

void* Device::Dump(gna_model_id modelId, gna_device_generation deviceGeneration, intel_gna_model_header* modelHeader, intel_gna_status_t* status, intel_gna_alloc_cb customAlloc)
{
    // Validate parameters

    Expect::NotNull(status);
    Expect::NotNull(modelHeader);
    Expect::NotNull((void *)customAlloc);
    Expect::Equal(GNA_1_0_EMBEDDED, deviceGeneration, GNA_CPUTYPENOTSUPPORTED); // Temporary limitation

    FakeDetector detector{ *ioctlSender, FakeDetector::GetDeviceVersion(deviceGeneration) };

    auto memoryId = getMemoryId(modelId);
    auto totalMemory = getMemory(memoryId);
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
    
    // copying data..
    void *data = totalMemory->Get() + totalMemory->InternalSize;
    void *dumpData = static_cast<uint8_t*>(address) + internalSize;
    memcpy(dumpData, data, totalMemory->ModelSize);

    // TODO:3: review
    // filling model header
    auto const &input = model.GetLayer(0)->Input;
    auto const &output = model.GetLayer(layerCount - 1)->Output;
    uint32_t outputsOffset = hwModel->GetOutputOffset(layerCount - 1);
    uint32_t inputsOffset = hwModel->GetInputOffset(0);
    *modelHeader = { 0, static_cast<uint32_t>(dumpedModelTotalSize), 1, layerCount, input.Mode.Size,
        output.Mode.Size, input.Count, output.Count, inputsOffset, outputsOffset };

    return address;
}
