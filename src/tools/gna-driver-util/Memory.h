/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "common.h"

#include <memory>
#include <stdexcept>

#if defined(__GNUC__)
#include <cstring>
#include <mm_malloc.h>
#endif

class Memory
{
    void* buffer = nullptr;
    uint64_t id = 0;
    uint32_t size = 0;

public:
    Memory(uint32_t userSize)
    {
        buffer = _mm_malloc(userSize, PAGE_SIZE);
        buffer != nullptr ? memset(buffer, 0, userSize) : throw std::runtime_error("buffer is null");
        size = userSize;
    }

    ~Memory()
    {
        id = 0;
        size = 0;
        if (buffer != nullptr)
        {
            _mm_free(buffer);
            buffer = nullptr;
        }
    }
    void* GetBuffer() const
    {
        return buffer;
    }

    uint32_t GetSize() const
    {
        return size;
    }

    uint64_t GetId() const
    {
        return id;
    }

    void SetId(uint64_t requestedId)
    {
        id = requestedId;
    }

    void copy(const uint8_t * src)
    {
        memcpy(buffer, src, sizeof(int8_t) * 128);
    }
};
