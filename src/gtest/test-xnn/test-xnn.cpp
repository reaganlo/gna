#include <chrono>
#include <iostream>

#include "gtest/gtest.h"

std::string currentDateTime() {
    return std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}

class CnFirstTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        m_string = currentDateTime();
    }
    std::string m_string;
};

TEST_F(CnFirstTest, Test1) {
    std::cout << m_string << std::endl;
}

TEST_F(CnFirstTest, Test2) {
    std::cout << m_string << std::endl;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    return RUN_ALL_TESTS();
}