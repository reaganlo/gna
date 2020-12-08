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
#include "gna2-memory-impl.h"
#include "Macros.h"
#include "Memory.h"

#include <algorithm>

using namespace GNA;

MemoryContainerElement::MemoryContainerElement(Memory const& memoryIn, uint32_t notAlignedIn, uint32_t pageAlignedIn) :
    std::reference_wrapper<Memory const>{memoryIn},
    NotAligned{ notAlignedIn },
    PageAligned{ pageAlignedIn }
{
}

Memory const* MemoryContainerElement::operator->() const
{
    return &get();
}

void MemoryContainer::Append(MemoryContainer const & source)
{
    for (auto const & value : source)
    {
        Emplace(value);
    }
}

void MemoryContainer::Emplace(Memory const & value)
{
    if (!Contains(value, value.GetSize()))
    {
        emplace_back(value, totalMemorySize, totalMemorySizeAlignedToPage);
        totalMemorySizeAlignedToPage += RoundUp(value.GetSize(), MemoryBufferAlignment);
        totalMemorySize += value.GetSize();
    }
}

MemoryContainer::const_iterator MemoryContainer::FindByAddress(BaseAddress const& address) const
{
    auto const foundIt = std::find_if(cbegin(), cend(),
        [&address](auto const & memory)
    {
        return address.InRange(memory->GetBuffer(), memory->GetSize());
    });

    return foundIt;
}

bool MemoryContainer::Contains(const void* buffer, const size_t bufferSize) const
{
    auto const & memory = FindByAddress(buffer);
    return cend() != memory &&
        Expect::InMemoryRange(buffer, bufferSize,
        (*memory)->GetBuffer(), (*memory)->GetSize());
}

uint32_t MemoryContainer::GetBufferOffset(const BaseAddress& address, uint32_t alignment, uint32_t initialOffset) const
{
    auto const foundIt = FindByAddress(address);
    if (cend() != foundIt)
    {
        auto const internalOffset = address.GetOffset(BaseAddress{ (*foundIt)->GetBuffer() });
        if (1 == alignment)
        {
            return initialOffset + foundIt->NotAligned + internalOffset;
        }
        if (MemoryBufferAlignment == alignment)
        {
            return initialOffset + foundIt->PageAligned + internalOffset;
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
        auto const memorySize = memory->GetSize();
        auto const memoryBuffer = memory->GetBuffer();
        memcpy_s(address, destinationSize, memoryBuffer, memorySize);
        destinationSize -= memorySize;
        address += memorySize;
    }
}
