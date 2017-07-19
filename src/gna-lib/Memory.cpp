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

#include "Memory.h"

#include "CompiledModel.h"
#include "Validator.h"

using namespace GNA;

// just makes object from arguments
Memory::Memory(void * bufferIn, const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount) :
    Address{bufferIn},
    InternalSize{CompiledModel::CalculateInternalModelSize(layerCount, gmmCount)},
    ModelSize{ALIGN64(userSize)},
    size{CompiledModel::CalculateModelSize(userSize, layerCount, gmmCount)}
{};

// allocates and zeros memory
Memory::Memory(const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount) :
    InternalSize{CompiledModel::CalculateInternalModelSize(layerCount, gmmCount)},
    ModelSize{ALIGN64(userSize)},
    size{CompiledModel::CalculateModelSize(userSize, layerCount, gmmCount)}
{
    Expect::True(size > 0, GNA_INVALIDMEMSIZE);
    buffer = _gna_malloc(size);
    Expect::ValidBuffer(buffer);
    memset(buffer, 0, size);
};

Memory::~Memory()
{
    if (buffer)
    {
        _gna_free(buffer);
        buffer = nullptr;
        size = 0;
    }
}

void Memory::Map(gna_model_id model_id)
{
    if (mapped)
        throw GnaException(GNA_UNKNOWN_ERROR);

    OVERLAPPED notifyOverlapped;
    sender.IoctlSendEx(GNA_IOCTL_NOTIFY, nullptr, 0, nullptr, 0, &notifyOverlapped);

    modelId = static_cast<uint64_t>(model_id);

    // write model id in user buffer
    // driver will retrieve it
    *reinterpret_cast<uint64_t*>(buffer) = static_cast<uint64_t>(modelId);

    sender.IoctlSend(
        GNA_IOCTL_MEM_MAP,
        nullptr,
        0,
        buffer,
        size,
        TRUE);

    sender.WaitOverlapped(&notifyOverlapped);

    mapped = true;
}

void Memory::Unmap()
{
    if (!mapped)
        throw GnaException(GNA_UNKNOWN_ERROR);

    sender.IoctlSend(GNA_IOCTL_MEM_UNMAP, &modelId, sizeof(modelId), nullptr, 0);

    mapped = false;
}
