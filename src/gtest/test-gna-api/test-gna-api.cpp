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

#include "gna-api.h"
#include "../../gna-api/gna2-model-api.h"

#include <chrono>
#include <gtest/gtest.h>
#include <iostream>

#define UNREFERENCED_PARAMETER(P) (P)

class TestGnaApi : public testing::Test
{
protected:
    int timeout = 300;
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

};

class TestGnaModelApi : public TestGnaApi
{
protected:
    static void * Allocator(uint32_t size)
    {
        return malloc(size);
    }

    static void * InvalidAllocator(uint32_t size)
    {
        UNREFERENCED_PARAMETER(size);
        return nullptr;
    }
};

TEST_F(TestGnaApi, allocateMemory)
{
    uint32_t sizeRequested = 47;
    uint32_t sizeGranted = 0;
    void * mem = nullptr;
    gna_status_t status;
    status = GnaAlloc(sizeRequested, &sizeGranted, &mem);
    EXPECT_LE(sizeRequested, sizeGranted);
    ASSERT_EQ(status, GNA_SUCCESS);
    GnaFree(mem);
}

TEST_F(TestGnaModelApi, Gna2DataTypeGetSizeSuccesfull)
{
    auto size = Gna2DataTypeGetSize(Gna2DataTypeVoid);
    ASSERT_EQ(size, static_cast<uint32_t>(0));
    size = Gna2DataTypeGetSize(Gna2DataTypeBoolean);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt4);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt8);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt16);
    ASSERT_EQ(size, static_cast<uint32_t>(2));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt32);
    ASSERT_EQ(size, static_cast<uint32_t>(4));
    size = Gna2DataTypeGetSize(Gna2DataTypeInt64);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint4);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint8);
    ASSERT_EQ(size, static_cast<uint32_t>(1));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint16);
    ASSERT_EQ(size, static_cast<uint32_t>(2));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint32);
    ASSERT_EQ(size, static_cast<uint32_t>(4));
    size = Gna2DataTypeGetSize(Gna2DataTypeUint64);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypeCompoundBias);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypePwlSegment);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
    size = Gna2DataTypeGetSize(Gna2DataTypeWeightScaleFactor);
    ASSERT_EQ(size, static_cast<uint32_t>(8));
}

TEST_F(TestGnaModelApi, Gna2DataTypeGetSizeIncorrectType)
{
    auto size = Gna2DataTypeGetSize(static_cast<Gna2DataType>(Gna2DataTypeInt8 - 100));
    ASSERT_EQ(size, static_cast<uint32_t>(GNA2_NOT_SUPPORTED));
}

TEST_F(TestGnaModelApi, Gna2ModelCreateNull)
{
    const auto status = GnaModelCreate(0, nullptr, nullptr);
    ASSERT_NE(GNA_SUCCESS, status);
}

TEST_F(TestGnaModelApi, Gna2ModelCreate2Successfull)
{
    uint32_t modelId = 0;
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(0, &model, &modelId);
    ASSERT_TRUE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, DISABLED_Gna2ModelCreate2NullModel)
{
    uint32_t modelId = 0;
    const auto status = Gna2ModelCreate(0, nullptr, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, DISABLED_Gna2ModelCreate2NullModelId)
{
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(0, &model, nullptr);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ItemTypeModelOperationsodelCreate2InvalidDeviceIndex)
{
    uint32_t modelId = 0;
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(100, &model, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ModelOperationInitSuccessfull)
{
    struct Gna2Operation operation = {};
    const auto type = Gna2OperationTypeCopy;
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    ASSERT_TRUE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ModelOperationInitNullOperation)
{
    const auto type = static_cast<Gna2OperationType>(100);
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(nullptr,
        type, userAllocator);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ModelOperationInitInvalidType)
{
    struct Gna2Operation operation = {};
    const auto type = static_cast<Gna2OperationType>(100);
    const auto userAllocator = &Allocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}


TEST_F(TestGnaModelApi, Gna2ModelOperationInitNullAllocator)
{
    struct Gna2Operation operation = {};
    const auto type = Gna2OperationTypeCopy;
    const auto status = Gna2ModelOperationInit(&operation,
        type, nullptr);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaModelApi, Gna2ModelOperationInitInvalidAllocator)
{
    struct Gna2Operation operation = {};
    const auto type = Gna2OperationTypeCopy;
    const auto userAllocator = &InvalidAllocator;
    const auto status = Gna2ModelOperationInit(&operation,
        type, userAllocator);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
