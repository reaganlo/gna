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

#include "DriverInterface.h"
#include "GnaUtilConfig.h"
#include "Memory.h"

#ifdef WIN32
#include "WindowsDriverInterface.h"
#else
#include "LinuxDriverInterface.h"
#endif


#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#include <cstring>
#endif

#include <iostream>
#include <stdio.h>
#include <thread>
#include <array>

#define RtlZeroMemory(Destination,Length) memset((Destination),0,(Length))
#define ZeroMemory RtlZeroMemory

int16_t* calculateOutputOffset(Memory& dataMemory, const GnaUtilConfig& config)
{
    auto buf_size_weights = Gna2RoundUpTo64(config.dataSize.weights); // note that buffer alignment to 64-bytes is required by GNA HW
    auto buf_size_inputs = Gna2RoundUpTo64(config.dataSize.input);
    auto buf_size_biases = Gna2RoundUpTo64(config.dataSize.biases);
    auto buf_size_outputs = Gna2RoundUpTo64(config.dataSize.output);

    auto pinned_mem_ptr = (uint8_t*)dataMemory.GetBuffer();
    auto pinned_weights = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_weights, config.modelData.weights.get(),config.dataSize.weights);
    pinned_mem_ptr += buf_size_weights;

    auto pinned_inputs = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_inputs, config.modelData.input.get(),config.dataSize.input);
    pinned_mem_ptr += buf_size_inputs;  // fast-forwards current pinned memory pointer to the next free block

    auto pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, config.modelData.biases.get(), config.dataSize.biases);// puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases; // fast-forwards current pinned memory pointer to the next free block

    auto pinned_outputs = (int16_t*)pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs; // fast-forwards the current pinned memory pointer by the space needed for outputs

    return pinned_outputs;
}

void displayResults(int32_t* pinned_outputs, const GnaUtilConfig& config)
{
    const auto numberOfColumns = 4;
    auto out = pinned_outputs;

    std::cout << "Results: " << std::endl;
    auto nOutputs = config.dataSize.output/sizeof(uint32_t);

    for (unsigned int i = 0; i < nOutputs; ++i)
    {
        std::cout << *(out + i) << " ";
        if (i % numberOfColumns == numberOfColumns - 1)
        {
            std::cout << std::endl;
        }
    }
}

static int computeModelOnHardware(DriverInterface& di, const GnaUtilConfig &config)
{
    auto allocateMemoryForData = config.dataSize.input;
    allocateMemoryForData += config.dataSize.weights;
    allocateMemoryForData += config.dataSize.biases;
    allocateMemoryForData += config.dataSize.output;

    std::cout << "Computed model memory: " << allocateMemoryForData
        << std::endl;

    auto dataMemory = Memory(allocateMemoryForData);
    auto descriptorMemory = Memory(config.dataSize.descriptor);

    auto driverBufferDescriptor(descriptorMemory);

    auto pinnedOutputs = calculateOutputOffset(dataMemory, config);

    HardwareRequest hardwareRequest;
    hardwareRequest.DriverMemoryObjects.push_back(driverBufferDescriptor);

    auto dataMemoryId = di.MemoryMap(dataMemory.GetBuffer(),
        dataMemory.GetSize());
    auto descriptorMemoryId = di.MemoryMap(descriptorMemory.GetBuffer(),
        descriptorMemory.GetSize());

    try {
        dataMemory.SetId(dataMemoryId);
        descriptorMemory.SetId(descriptorMemoryId);
        descriptorMemory.copy((uint8_t*)config.modelData.descriptor.get());

        di.Submit(hardwareRequest, config);

        displayResults((int32_t*)pinnedOutputs, config);
    }
    catch (...)
    {
        // TODO handle properly
        std::cout << "Exception happened\n";
    }

    di.MemoryUnmap(dataMemory.GetId());
    di.MemoryUnmap(descriptorMemory.GetId());
    return 0;
}

int main(int argc, char** argv)
{
    GnaUtilConfig file(argc, argv);
#if _WIN32
    WindowsDriverInterface di;
#else
    LinuxDriverInterface di;
#endif

    bool found = 0;
    int computeResult = -1;

    for (uint32_t i = 0; i < DriverInterface::MAX_GNA_DEVICES; ++i)
    {
        auto deviceFound = di.discoverDevice(i);
        if (deviceFound == false)
        {
            continue;
        }
        found = true;
        std::cout << "Device has been found" << std::endl;

        try
        {
            std::cout << "Computing model" << std::endl;
            computeResult = computeModelOnHardware(di, file);
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }
        break;
    }

    if (found == true && computeResult == 0)
    {
        std::cout << "Program ended successfully" << std::endl;
    }
    else if (found == false)
    {
        std::cerr << "Device not found" << std::endl;
    }
    else
    {
        std::cerr << "Model could not  run" << std::endl;
    }

    return 0;
}
