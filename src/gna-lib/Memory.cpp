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

#include <cstring>

#include "AccelerationDetector.h"
#include "CompiledModel.h"
#include "Validator.h"

using namespace GNA;

// just makes object from arguments
Memory::Memory(void * bufferIn, const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount, IoctlSender &sender) :
    Address{bufferIn},
    InternalSize{CompiledModel::CalculateInternalModelSize(layerCount, gmmCount)},
    ModelSize{ALIGN64(userSize)},
    size{CompiledModel::CalculateModelSize(userSize, layerCount, gmmCount)},
    ioctlSender{sender}
{};

// allocates and zeros memory
Memory::Memory(const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount, IoctlSender &sender) :
    InternalSize{CompiledModel::CalculateInternalModelSize(layerCount, gmmCount)},
    ModelSize{ALIGN64(userSize)},
    size{CompiledModel::CalculateModelSize(userSize, layerCount, gmmCount)},
    ioctlSender{sender}
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

void Memory::Map()
{
    if (mapped)
    {
        throw GnaException(GNA_ERR_MEMORY_ALREADY_MAPPED);
    }

    id = ioctlSender.MemoryMap(buffer, size);

    mapped = true;
}

void Memory::Unmap()
{
    if (!mapped)
    {
        throw GnaException(GNA_ERR_MEMORY_ALREADY_UNMAPPED);
    }

    ioctlSender.MemoryUnmap(id);
    mapped = false;
}

uint64_t Memory::GetId() const
{
    if (!mapped)
    {
        throw GnaException(GNA_ERR_MEMORY_NOT_MAPPED);
    }

    return id;
}

void Memory::AllocateModel(const gna_model_id modelId, const gna_model *model, const AccelerationDetector& detector)
{
    void * descriptorsBase = Get() + descriptorsSize;

    auto modelInternalSize = CompiledModel::CalculateInternalModelSize(model);
    if (descriptorsSize + modelInternalSize > InternalSize)
    {
        throw GnaException(GNA_ERR_RESOURCES);
    }

    modelDescriptors[modelId] = descriptorsBase;
    descriptorsSize += modelInternalSize;

    models[modelId] = createModel(modelId, model, detector);
}

void Memory::DeallocateModel(gna_model_id modelId)
{
    models[modelId].reset();
}

CompiledModel& Memory::GetModel(gna_model_id modelId)
{
    try
    {
        auto& model = models.at(modelId);
        return *model.get();
    }
    catch (const std::out_of_range& e)
    {
        throw GnaException(GNA_INVALID_MODEL);
    }
}

void * Memory::GetDescriptorsBase(gna_model_id modelId) const
{
    return modelDescriptors.at(modelId);
}

std::unique_ptr<CompiledModel> Memory::createModel(const gna_model_id modelId, const gna_model *model,
    const AccelerationDetector &detector)
{
    return std::make_unique<CompiledModel>(modelId, model, *this, ioctlSender, detector);
}
