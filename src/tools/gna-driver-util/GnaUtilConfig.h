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

#include <cstdint>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

class GnaUtilConfig
{
private:
    struct Data
    {
        std::unique_ptr<char[]> input;
        std::unique_ptr<char[]> weights;
        std::unique_ptr<char[]> biases;
        std::unique_ptr<char[]> descriptor;
    };

    struct DataSizes
    {
        uint32_t input = 0;
        uint32_t weights = 0;
        uint32_t biases = 0;
        uint32_t output = 0;
        uint32_t descriptor = 0;
    };

    struct Config
    {
        uint32_t bufferOffset;
        uint32_t patchOffset;
        uint8_t* inference;
    };

    const int argumentCount = 3;

public:
    DataSizes dataSize;
    Data modelData;
    Config modelConfig;

    uint64_t inferenceConfigSize;

    GnaUtilConfig(int argc, char** argv);

    void readData(char* argv);

    void readConfig(char* argv);

    GnaUtilConfig(const GnaUtilConfig & utilConfig) = delete;

    GnaUtilConfig& operator = (const GnaUtilConfig & rhs) = delete;

    GnaUtilConfig(GnaUtilConfig && rhs) = delete;

    GnaUtilConfig& operator = (GnaUtilConfig && utilConfig) = delete;
};
