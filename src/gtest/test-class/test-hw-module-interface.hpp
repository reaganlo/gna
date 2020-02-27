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

#pragma once

#include "HwModuleInterface.hpp"

#include "GnaException.h"
#include "gna2-model-api.h"
#include "DataMode.h"

#include "gtest/gtest.h"

class HwModuleInterfaceTest : public testing::Test
{
protected:
    void SetUp() override
    {}

    void ExpectGnaException(void* arg1, void* arg2, bool is1D)
    {
        hwModule = GNA::HwModuleInterface::Create("gna_hw");
        ASSERT_FALSE(!hwModule);
        auto const cnnIn =
            reinterpret_cast<GNA::ConvolutionFunction2D const*>(arg1);
        auto const poolingIn =
            reinterpret_cast<GNA::PoolingFunction2D const*>(arg2);
        EXPECT_THROW(
            hwModule->GetCnnParams(cnnIn, poolingIn, GNA::DataMode{ Gna2DataTypeInt8 }, is1D),
            GNA::GnaException);
    }

    std::unique_ptr<GNA::HwModuleInterface const> hwModule;
};

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
