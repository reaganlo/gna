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

#include "GnaUtilConfig.h"

#include <array>

void GnaUtilConfig::readData(char* argv)
{
    std::string dataFileName(argv);
    std::ifstream dataFile(dataFileName, std::ios::binary | std::ios::in);
    if (dataFile.is_open())
    {
        dataFile.read((char*)&dataSize, sizeof(DataSizes));
        modelData.input = std::make_unique<char[]>(dataSize.input);
        modelData.weights = std::make_unique<char[]>(dataSize.weights);
        modelData.biases = std::make_unique<char[]>(dataSize.biases);
        modelData.descriptor = std::make_unique<char[]>(dataSize.descriptor);

        dataFile.read(modelData.input.get(), dataSize.input);
        dataFile.read(modelData.weights.get(), dataSize.weights);
        dataFile.read(modelData.biases.get(), dataSize.biases);
        dataFile.read(modelData.descriptor.get(), dataSize.descriptor);

        if (dataFile.rdstate())
        {
            std::runtime_error("File with model could not be found!");
            return;
        }

        dataFile.close();
        std::cout << "Data file has been read" << std::endl;
    }
    else
    {
        std::runtime_error("File with model could not be found!");
    }
}

void GnaUtilConfig::readConfig(char* argv)
{
    std::string configFileName(argv);
    std::ifstream configFile(configFileName, std::ios::binary | std::ios::in);
    if (configFile.is_open())
    {
        configFile.read((char*)&inferenceConfigSize,
            sizeof(inferenceConfigSize));
        configFile.read((char*)&modelConfig.bufferOffset,
            sizeof(modelConfig.bufferOffset));
        configFile.read((char*)&modelConfig.patchOffset,
            sizeof(modelConfig.patchOffset));

        modelConfig.inference = new uint8_t[inferenceConfigSize];

        configFile.read((char*)modelConfig.inference, (std::streamsize)inferenceConfigSize);

        configFile.close();
        std::cout << "Config file has been read" << std::endl;
    }
    else
    {
        std::runtime_error("File with model could not be found!");
    }
}

GnaUtilConfig::GnaUtilConfig(int argc, char** argv)
{
    if (argc == argumentCount)
    {
        readData(argv[1]);
        readConfig(argv[2]);
    }
    else
    {
        std::cout << "\nParameters order\n" << std::endl;
        std::cout << "=====================================\n" << std::endl;
        std::cout << "gna-driver-util data_file config_file\n" << std::endl;
    }
}
