/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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


#include "test-hw-module-interface.hpp"

#include "DataMode.h"

using namespace GNA;

void HwModuleInterfaceTest::ExpectGnaException(void* arg1, void* arg2, bool is1D)
{
    hwModule = GNA::HwModuleInterface::Create("gna_hw");
    ASSERT_FALSE(!hwModule);
    auto const cnnIn =
        reinterpret_cast<GNA::ConvolutionFunction2D const*>(arg1);
    auto const poolingIn =
        reinterpret_cast<GNA::PoolingFunction2D const*>(arg2);
    EXPECT_THROW(
        hwModule->GetCnnParams(cnnIn, poolingIn, GNA::DataMode(Gna2DataTypeInt8), is1D),
        GNA::GnaException);
}

TEST_F(HwModuleInterfaceTest, CreateNullName)
{
    EXPECT_THROW(
        hwModule = HwModuleInterface::Create(nullptr),
        GnaException);
    ASSERT_TRUE(!hwModule);
}

TEST_F(HwModuleInterfaceTest, CreateEmptyName)
{
    EXPECT_THROW(
        hwModule = HwModuleInterface::Create(nullptr),
        GnaException);
    ASSERT_TRUE(!hwModule);
}

TEST_F(HwModuleInterfaceTest, CreateNonExistingName)
{
    hwModule = HwModuleInterface::Create("xxx");
    ASSERT_FALSE(!hwModule);
    EXPECT_THROW(
        hwModule->GetCnnParams(nullptr, nullptr, DataMode{ Gna2DataTypeInt8 }, false),
        GnaException);
}

#if 1 == GNA_HW_LIB_ENABLED

TEST_F(HwModuleInterfaceTest, CreateSuccessful)
{
    hwModule = HwModuleInterface::Create("gna_hw");
    ASSERT_FALSE(!hwModule);
}

TEST_F(HwModuleInterfaceTest, NullArgument1_1D)
{
    ExpectGnaException(nullptr, reinterpret_cast<void*>(0x3452344), true);
}

TEST_F(HwModuleInterfaceTest, NullArgument2_1D)
{
    ExpectGnaException(reinterpret_cast<void*>(0x3452344), nullptr, true);
}

TEST_F(HwModuleInterfaceTest, NullArgument1and2_1D)
{
    ExpectGnaException(nullptr, nullptr, true);
}

TEST_F(HwModuleInterfaceTest, NullArgument1_2D)
{
    ExpectGnaException(nullptr, reinterpret_cast<void*>(0x3452344), false);
}

TEST_F(HwModuleInterfaceTest, NullArgument2_2D)
{
    ExpectGnaException(reinterpret_cast<void*>(0x3452344), nullptr, false);
}

TEST_F(HwModuleInterfaceTest, NullArgument1and2_2D)
{
    ExpectGnaException(nullptr, nullptr, false);
}
#else

TEST_F(HwModuleInterfaceTest, NotImplemented_1D)
{
    ExpectGnaException(reinterpret_cast<void*>(0x3452344), reinterpret_cast<void*>(0x3452344), true);
}

TEST_F(HwModuleInterfaceTest, NotImplemented_2D)
{
    ExpectGnaException(reinterpret_cast<void*>(0x3452344), reinterpret_cast<void*>(0x3452344), false);
}

#endif
