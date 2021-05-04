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

#include "gna2-common-api.h"
#include "gna2-model-api.h"

#include "gna2-common-impl.h"

#include "ModelError.h"

#include "gtest/gtest.h"
#include <map>
#include <string>

class TestInternal : public ::testing::Test
{
public:
    TestInternal(){}
    TestInternal(const TestInternal& rhs) = delete;

protected:
    template<class T>
    void TestStringMapLength(const uint32_t maxLenFromApi, const std::map<T, std::string> & container)
    {
        for (const auto& item : container)
        {
            // any message should be longer than 1024
            ASSERT_LT(item.second.size(), 1024);
            ASSERT_LT(item.second.size(), maxLenFromApi);
        }
    }
};

TEST_F(TestInternal, Gna2StatusToStringMap_desc_sizes)
{
    TestStringMapLength(
        Gna2StatusGetMaxMessageLength(),
        GNA::StatusHelper::GetStringMap());
}

TEST_F(TestInternal, Gna2ErrorTypeToStringMap_desc_sizes)
{
    TestStringMapLength(
        Gna2ErrorTypeGetMaxMessageLength(),
        GNA::ModelErrorHelper::GetAllErrorTypeStrings());
}
