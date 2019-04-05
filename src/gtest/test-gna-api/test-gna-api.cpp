#include <iostream>
#include <chrono>
#include <gtest/gtest.h>
#include "gna-api.h"

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
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}