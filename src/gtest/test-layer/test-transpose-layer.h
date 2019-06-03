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

#include "test-layer.h"

#include "HardwareCapabilities.h"
#include "Validator.h"

#include "common.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace GNA;

class TestTransposeLayer : public ::testing::Test
{
public:
    TestTransposeLayer() :
        emptyValidator
        {
            HardwareCapabilities(),
            ValidBoundariesFunctor { [] (const void *buffer, size_t bufferSize) {} }
        }
    {
        alignedInput = (int16_t *)_kernel_malloc(sizeof(*alignedInput) * numberOfElements);
        if (!alignedInput)
        {
            throw;
        }

        alignedOutput = (int16_t *) _kernel_malloc(sizeof(*alignedOutput) * numberOfElements);
        if (!alignedOutput)
        {
            _gna_free(alignedInput);
            alignedInput = nullptr;
            throw;
        }
    }

    ~TestTransposeLayer()
    {
        if (alignedInput != nullptr)
        {
            _gna_free(alignedInput);
            alignedInput = nullptr;
        }

        if (alignedOutput != nullptr)
        {
            _gna_free(alignedOutput);
            alignedOutput = nullptr;
        }
    }

    TestTransposeLayer(const TestTransposeLayer& rhs) = delete;

    void VerifyOutputs(const int16_t *output, const int16_t *refOutput)
    {
        for (uint32_t i = 0; i < numberOfElements; i++)
        {
            EXPECT_EQ(output[i], refOutput[i]);
        }
    }

    const BaseValidator emptyValidator;

    static constexpr uint16_t numberOfElements = 32;
    static const int16_t flatInput[numberOfElements];
    static const int16_t interleaveInput[numberOfElements];

    int16_t *alignedInput = nullptr;
    int16_t *alignedOutput = nullptr;
};


