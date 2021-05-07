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
