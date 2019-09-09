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

#include "MemoryContainer.h"

#include "Expect.h"
#include "GnaException.h"
#include "Memory.h"

#include <algorithm>
#include <string.h>

using namespace GNA;

void MemoryContainer::Append(MemoryContainer const & source)
{
    for (auto const & value : source)
    {
        Emplace(value.second);
    }
}

void MemoryContainer::Emplace(Memory const & value)
{
    if (!Contains(value, value.GetSize()))
    {
        offsets.emplace(size(), std::make_pair(totalMemorySize, totalMemorySizeAlignedToPage));
        emplace(size(), value);
        totalMemorySizeAlignedToPage += RoundUp(value.GetSize(), PAGE_SIZE);
        totalMemorySize += value.GetSize();
    }
}

MemoryContainer::iterator MemoryContainer::erase(const_iterator where)
{
    auto const size = where->second.GetSize();
    auto const & erased = MemoryContainerType::erase(where);
    if (where != cend())
    {
        totalMemorySizeAlignedToPage -= RoundUp(size, PAGE_SIZE);
        totalMemorySize -= size;
        invalidateOffsets();
    }
    return erased;
}

MemoryContainer::size_type MemoryContainer::erase(key_type const& key)
{
    auto const & where = find(key);
    erase(where);
    return 1;
}

MemoryContainer::const_iterator MemoryContainer::FindByAddress(BaseAddress const& address) const
{
    auto const foundIt = std::find_if(cbegin(), cend(),
        [&address](auto const & memory)
    {
        return address.InRange(memory.second.GetBuffer(), memory.second.GetSize());
    });

    return foundIt;
}

bool MemoryContainer::Contains(const void* buffer, const size_t bufferSize) const
{
    auto const & memory = FindByAddress(buffer);
    if (cend() != memory &&
        Expect::InMemoryRange(buffer, bufferSize, memory->second.GetBuffer(), memory->second.GetSize()))
    {
        return true;
    }
    return false;
}

uint32_t MemoryContainer::GetBufferOffset(const BaseAddress& address, uint32_t alignment, uint32_t initialOffset) const
{
    auto const foundIt = FindByAddress(address);
    if (cend() != foundIt)
    {
        auto const internalOffset = address.GetOffset(BaseAddress{ foundIt->second.GetBuffer() });
        auto const containerOffset = offsets.at(foundIt->first);
        if (1 == alignment)
        {
            return initialOffset + containerOffset.first + internalOffset;;
        }
        if (PAGE_SIZE == alignment)
        {
            return initialOffset + containerOffset.second + internalOffset;
        }
    }
    return 0;
}

void MemoryContainer::CopyData(void* destination, size_t destinationSize) const
{
    if (destinationSize < totalMemorySize)
    {
        throw GnaException{ Gna2StatusResourceAllocationError };
    }

    auto address = static_cast<uint8_t *>(destination);
    for (const auto & memory : *this)
    {
        auto const memorySize = memory.second.GetSize();
        auto const memoryBuffer = memory.second.GetBuffer();
        memcpy_s(address, destinationSize, memoryBuffer, memorySize);
        destinationSize -= memorySize;
        address += memorySize;
    }
}

void MemoryContainer::WriteData(FILE* file) const
{
    Expect::NotNull(file);
    for (const auto & memory : *this)
    {
        auto const memorySize = memory.second.GetSize();
        auto const memoryBuffer = memory.second.GetBuffer();
        fwrite(memoryBuffer, memorySize, sizeof(uint8_t), file);
    }
}

void MemoryContainer::invalidateOffsets()
{
    offsets.clear();
    uint32_t offset = 0;
    uint32_t offsetPageAligned = 0;
    //for (auto && memoryIter = cbegin(); memoryIter != cend(); ++memoryIter)
    for (auto const & memoryIter : *this)
    {
        offsets[memoryIter.first] = std::make_pair(offset, offsetPageAligned);

        auto const memorySize = memoryIter.second.GetSize();
        offset += memorySize;
        offsetPageAligned += RoundUp(memorySize, PAGE_SIZE);
    }
}
