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

TEST_F(TestGnaApi, nullModelCreate)
{
    auto status = GnaModelCreate(0, nullptr, nullptr);
    ASSERT_NE(GNA_SUCCESS, status);
}

TEST_F(TestGnaApi, ModelCreate2Successfull)
{
    uint32_t modelId = 0;
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(0, &model, &modelId);
    ASSERT_TRUE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaApi, DISABLED_ModelCreate2NullModel)
{
    uint32_t modelId = 0;
    const auto status = Gna2ModelCreate(0, nullptr, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaApi, DISABLED_ModelCreate2NullModelId)
{
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(0, &model, nullptr);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

TEST_F(TestGnaApi, ModelCreate2InvalidDeviceIndex)
{
    uint32_t modelId = 0;
    Gna2Model model = {};
    const auto status = Gna2ModelCreate(100, &model, &modelId);
    ASSERT_FALSE(Gna2StatusIsSuccessful(status));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
