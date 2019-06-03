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

#include "Validator.h"

#include "KernelArguments.h"

#include "gna-api-types-xnn.h"

#include <gtest/gtest.h>

#include <stdint.h>
#include <stdlib.h>

using namespace GNA;

void *GnaMalloc(uint32_t size);

class TestLayer : public ::testing::Test
{
public:
    TestLayer();
    TestLayer(const TestLayer& rhs) = delete;

    static KernelBuffers kernelBuffers;

    static void samplePwl(intel_pwl_segment_t *segments, uint32_t numberOfSegments);

    template<typename T>
    static void VerifyOutputs(const T *output, const T *refOutput, uint32_t elementCount)
    {
        for (uint32_t i = 0; i < elementCount; i++)
        {
            EXPECT_EQ(output[i], refOutput[i]);
        }
    }

    const BaseValidator emptyValidator;
};

