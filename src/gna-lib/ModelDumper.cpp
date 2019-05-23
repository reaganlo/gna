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

#include "CompiledModel.h"
#include "DataMode.h"
#include "Expect.h"
#include "HardwareModelSue1.h"
#include "Layer.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Memory.h"

#include "gna-api-dumper.h"
#include "gna-api-status.h"
#include "gna-api.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>

using namespace GNA;

void* Device::Dump(gna_model_id modelId, gna_device_generation deviceGeneration, intel_gna_model_header* modelHeader, Gna2Status* status, intel_gna_alloc_cb customAlloc)
{
    // Validate parameters

    Expect::NotNull(status);
    Expect::NotNull(modelHeader);
    Expect::NotNull(reinterpret_cast<void *>(customAlloc));
    Expect::Equal(GNA_1_0_EMBEDDED, deviceGeneration, Gna2StatusAccelerationModeNotSupported); // Temporary limitation

    auto& model = models.at(modelId);
    auto const layerCount = model->LayerCount;
    auto const gmmCount = model->GmmCount;

    auto const ldSize = HardwareModelSue1::CalculateDescriptorSize(layerCount, gmmCount);
    auto const modelSize = model->CalculateSize();
    auto const totalSize = ldSize + modelSize;

    void * address = customAlloc(totalSize);

    Expect::NotNull(address);
    memset(address, 0, totalSize);

    auto dumpMemory = std::make_unique<Memory>(address, ldSize);

    // creating HW layer descriptors directly into dump memory
    auto hwModel = std::make_unique<HardwareModelSue1>(
                    model->GetLayers(), model->GmmCount, std::move(dumpMemory));
    hwModel->Build(model->GetModelMemoryList());

    // copying data..
    void *data = static_cast<uint8_t*>(address) + ldSize;
    model->CopyData(data, modelSize);

    // TODO:3: review
    // filling model header
    auto const &input = model->GetLayer(0)->Input;
    auto const &output = model->GetLayer(layerCount - 1)->Output;
    uint32_t outputsOffset = hwModel->GetOutputOffset(layerCount - 1);
    uint32_t inputsOffset = hwModel->GetInputOffset(0);
    *modelHeader = { 0, static_cast<uint32_t>(totalSize), 1, layerCount, input.Mode.Size,
        output.Mode.Size, input.Count, output.Count, inputsOffset, outputsOffset, 0, 0, 0, {} };

    *status = Gna2StatusSuccess;
    return address;
}
