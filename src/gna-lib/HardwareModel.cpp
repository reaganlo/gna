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

#include "GnaDrvApi.h"
#include "GnaException.h"
#include "HardwareModel.h"

using namespace GNA;

HardwareModel::HardwareModel(gna_model_id modId, const SoftwareModel& model, void *userMemory,
    size_t userMemorySize, uint32_t hwInBuffSize) :
    modelId(modId),
    hwInBufferSize(hwInBuffSize),
    hwDescriptor(userMemory)
{
    mapMemory(userMemory, userMemorySize);
    build(model.Layers);
}

HardwareModel::~HardwareModel()
{
    unmapMemory();
}

void HardwareModel::mapMemory(void *buffer, size_t bufferSize)
{
    if (memoryMapped)
        throw GnaException(GNA_ERR_UNKNOWN);

    // write model id in user buffer
    // driver will retrieve it
    *reinterpret_cast<uint64_t*>(buffer) = static_cast<uint64_t>(modelId);

    status_t status = GNA_SUCCESS;
    status = IoctlSend(
        GNA_IOCTL_MEM_MAP,
        nullptr,
        0,
        buffer,
        bufferSize,
        TRUE);

    if (GNA_SUCCESS != status)
        throw GnaException(status);

    memoryMapped = true;
}

void HardwareModel::unmapMemory()
{
    uint64_t mId = static_cast<uint64_t>(modelId);
    status_t status = GNA_SUCCESS;
    status = IoctlSend(GNA_IOCTL_MEM_UNMAP, &mId, sizeof(mId), nullptr, 0);

    if (GNA_SUCCESS != status)
        throw GnaException(status);

    memoryMapped = false;
}

void HardwareModel::build(const std::vector<std::unique_ptr<Layer>>& layers)
{
    auto layerIndex = 0ui32;
    for (auto& layer : layers)
    {
        XNN_LYR *layerDescriptor = reinterpret_cast<XNN_LYR*>(
            reinterpret_cast<uintptr_t>(hwDescriptor) + (layerIndex * sizeof(XNN_LYR)));

        Layers.push_back(HardwareLayer::Create(*layer, layerDescriptor, hwDescriptor, hwInBufferSize));
        ++layerIndex;
    }
}