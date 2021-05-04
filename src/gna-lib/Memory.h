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
#include "gna2-model-export-api.h"

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#endif
#include <cstdint>

namespace GNA
{

/** GNA main memory required alignment size */
constexpr auto MemoryBufferAlignment = uint32_t{ 0x1000 };

/** Allocator with alignment for HW data buffers */
#define _gna_malloc(a)    _mm_malloc(a, MemoryBufferAlignment)
/** Allocator with alignment for intrinsics */
#define _kernel_malloc(a) _mm_malloc(a, 0x40)
#define _gna_free(a)      _mm_free(a)

class DriverInterface;

class Memory : public BaseAddress
{
public:
    Memory() = default;

    // just makes object from arguments
    Memory(void * bufferIn, uint32_t userSize, uint32_t alignment = GNA_BUFFER_ALIGNMENT);

    // allocates and zeros memory
    Memory(const uint32_t userSize, uint32_t alignment = GNA_BUFFER_ALIGNMENT);

    virtual ~Memory();

    void Map(DriverInterface& ddi);
    void Unmap(DriverInterface& ddi);

    uint64_t GetId() const;

    uint32_t GetSize() const
    {
        return size;
    }

    template<class T = void> T * GetBuffer() const
    {
        return Get<T>();
    }

    void SetTag(uint32_t newTag);

    Gna2MemoryTag GetMemoryTag() const;

    static const uint32_t GNA_BUFFER_ALIGNMENT = 64;
    static constexpr uint32_t GNA_MAX_MEMORY_FOR_SINGLE_ALLOC = 1 << 28;

protected:
    uint64_t id = 0;

    uint32_t size = 0;

    uint32_t tag = 0;

    bool mapped = false;

    bool deallocate = true;
};

}
