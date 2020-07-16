/*
 INTEL CONFIDENTIAL
 Copyright 2019-2020 Intel Corporation.

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

#include "gna2-device-api.h"
#include "gna2-memory-api.h"

#include "common.h"
#include "Macros.h"

#include <array>
#include <chrono>
#include <gtest/gtest.h>

class TestGnaApi : public testing::Test
{
protected:
    int timeout = 60000;
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point stop;

    void SetUp() override
    {
        start = std::chrono::system_clock::now();
    }

    void TearDown() override
    {
        stop = std::chrono::system_clock::now();
        auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        EXPECT_LE(durationMs, timeout);
    }

    void setTimeoutInMs(int set)
    {
        timeout = set;
    };

    auto getTimeoutInMs()
    {
        return timeout;
    }

    static void ExpectMemEqual(const uint8_t* dump, uint32_t dumpSize, const uint8_t* ref, uint32_t refSize);
    void GetLibraryVersionTest(bool versionBufferValid, bool versionBufferSizeValid,
        Gna2Status expectedStatus, const std::string& expected0) const;
};

class TestGnaApiEx : public TestGnaApi
{
protected:
    static void * Allocator(uint32_t size)
    {
        return malloc(size);
    }
    static void Free(void * ptr)
    {
        return free(ptr);
    }

    static void * AlignedAllocator(uint32_t size)
    {
        return _mm_malloc(size, PAGE_SIZE);
    }
    static void AlignedFree(void * ptr)
    {
        return _mm_free(ptr);
    }

    static void * InvalidAllocator(uint32_t size)
    {
        UNREFERENCED_PARAMETER(size);
        return nullptr;
    }

    void SetUp() override
    {
        TestGnaApi::SetUp();
        DeviceIndex = 0;
        auto status = Gna2DeviceOpen(DeviceIndex);
        ASSERT_EQ(status, Gna2StatusSuccess);
    }

    void TearDown() override
    {
        TestGnaApi::TearDown();
        auto status = Gna2DeviceClose(DeviceIndex);
        ASSERT_EQ(status, Gna2StatusSuccess);
        DeviceIndex = UINT32_MAX;
    }

    uint32_t DeviceIndex = UINT32_MAX;
};

class TestGnaModel : public TestGnaApiEx
{
protected:
    const uint32_t DefaultMemoryToUse = 8192;
    void * gnaMemory = nullptr;

    static void AllocateGnaMemory(const uint32_t required, void* & grantedMemory)
    {
        uint32_t grantedSize = 0;
        ASSERT_EQ(grantedMemory, nullptr);
        const auto status = Gna2MemoryAlloc(required, &grantedSize, &grantedMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        ASSERT_NE(grantedMemory, nullptr);
        ASSERT_EQ(grantedSize, required);
    }

    void SetUp() override
    {
        TestGnaApiEx::SetUp();
        AllocateGnaMemory(DefaultMemoryToUse, gnaMemory);
    }

    void TearDown() override
    {
        TestGnaApiEx::TearDown();
        const auto status = Gna2MemoryFree(gnaMemory);
        ASSERT_EQ(status, Gna2StatusSuccess);
        gnaMemory = nullptr;
    }
};
