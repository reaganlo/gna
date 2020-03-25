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

#pragma once

#include "Address.h"

#include <cstdint>
#include <map>
#include <vector>

namespace GNA
{

class Memory;

class MemoryContainerElement
{
public:
    MemoryContainerElement(Memory const& memoryIn, uint32_t notAlignedIn, uint32_t pageAlignedIn);
    operator Memory const & () const;
    void * GetBuffer() const;
    uint32_t GetSize() const;
    uint32_t GetNotAligned() const;
    uint32_t GetPageAligned() const;
    void ResetOffsets(uint32_t notAlignedIn, uint32_t pageAlignedIn);
private:
    std::reference_wrapper<Memory const> memory;
    uint32_t notAligned;
    uint32_t pageAligned;
};

using MemoryContainerType = std::vector< MemoryContainerElement >;

class MemoryContainer : public MemoryContainerType
{
public:
    void Append(MemoryContainer const & source);

    void Emplace(Memory const & value);

    const_iterator FindByAddress(BaseAddress const & address) const;

    bool Contains(const void *buffer, const size_t bufferSize = 1) const;

    uint32_t GetMemorySize() const
    {
        return static_cast<uint32_t>(totalMemorySize);
    }

    uint32_t GetMemorySizeAlignedToPage() const
    {
        return static_cast<uint32_t>(totalMemorySizeAlignedToPage);
    }

    uint32_t GetBufferOffset(const BaseAddress& address, uint32_t alignment = 1, uint32_t initialOffset = 0) const;

    template<typename T>
    void CopyEntriesTo(std::vector<T> & destination) const;

    void CopyData(void * destination, size_t destinationSize) const;

    void WriteData(FILE *file) const;

protected:
    void invalidateOffsets();

    uint32_t totalMemorySizeAlignedToPage = 0;

    uint32_t totalMemorySize = 0;
};

template <typename T>
void MemoryContainer::CopyEntriesTo(std::vector<T> & destination) const
{
    for (auto const & memory : *this)
    {
        destination.emplace_back(memory);
    }
}

}
