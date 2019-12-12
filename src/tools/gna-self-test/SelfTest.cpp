//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material contains trade secrets and proprietary
// and confidential information of Intel or its suppliers and licensors.
// The Material is protected by worldwide copyright and trade secret laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed in any way
// without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//*****************************************************************************
#include "SelfTest.h"

#include "gna2-api.h"
#include "gna2-common-api.h"
#include "gna2-device-api.h"
#include "gna2-memory-api.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <map>

void LogGnaStatus(const Gna2Status &status, const char *what)
{
    const auto maxLen = Gna2StatusGetMaxMessageLength();
    std::vector<char> message(maxLen);
    Gna2StatusGetMessage(status, message.data(), maxLen);
    logger.Verbose("%s: %s\n", what, message.data());
}

void GnaSelfTest::HandleGnaStatus(const Gna2Status &status, const char *what) const
{
    LogGnaStatus(status, what);
    if (status != Gna2StatusSuccess)
    {
        Handle(GSTIT_GENERAL_GNA_NO_SUCCESS);
    }
}

SelfTestDevice::SelfTestDevice(const GnaSelfTest& gst) : gnaSelfTest{ gst }
{
    // open the device
    const auto status = Gna2DeviceOpen(deviceId);
    LogGnaStatus(status, "Gna2DeviceOpen");
    if (status != Gna2StatusSuccess)
    {
        GnaSelfTestLogger::Error("Gna2DeviceOpen FAILED\n");
        gnaSelfTest.Handle(GSTIT_DEVICE_OPEN_NO_SUCCESS);
    }
    else
    {
        deviceOpened = true;
    }
}

// obtains pinned memory shared with the device
uint8_t * SelfTestDevice::Alloc(const uint32_t bytesRequested)
{
    uint32_t sizeGranted;
    void * memory;
    const auto status = Gna2MemoryAlloc(bytesRequested, &sizeGranted, &memory);
    gnaSelfTest.HandleGnaStatus(status, "Gna2MemoryAlloc");

    logger.Verbose("Requested: %i Granted: %i Address: %p\n", (int)bytesRequested, (int)sizeGranted, memory);

    if (nullptr == memory)
    {
        GnaSelfTestLogger::Error("Gna2MemoryAlloc: Memory allocation FAILED\n");
        gnaSelfTest.Handle(GSTIT_GNAALLOC_MEM_ALLOC_FAILED);
    }
    allocated_mems.push_back(memory);
    return static_cast<uint8_t *>(memory);
}

void SelfTestDevice::SampleModelCreate(const SampleModelForGnaSelfTest& model)
{
    gnaModel.NumberOfOperations = 1; // number of hidden layers, using 1 for simplicity sake
    gnaModel.Operations = &gnaOperation;

    uint32_t buf_size_weights = Gna2RoundUpTo64(model.GetWeightsByteSize()); // note that buffer alignment to 64-bytes is required by GNA HW
    uint32_t buf_size_inputs = Gna2RoundUpTo64(model.GetInputsByteSize());
    uint32_t buf_size_biases = Gna2RoundUpTo64(model.GetBiasesByteSize());
    uint32_t buf_size_outputs = Gna2RoundUpTo64(model.GetRefScoresByteSize());

    // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs;
    //uint32_t bytes_granted;

    auto gnaBegin = Alloc(bytes_requested);

    int16_t * const pinned_weights = (int16_t *)gnaBegin;
    model.CopyWeights(pinned_weights);    // puts the weights into the pinned memory
    gnaBegin += buf_size_weights;   // fast-forwards current pinned memory pointer to the next free block

    pinned_inputs = (int16_t *)gnaBegin;
    model.CopyInputs(pinned_inputs);    // puts the inputs into the pinned memory
    gnaBegin += buf_size_inputs;  // fast-forwards current pinned memory pointer to the next free block

    int32_t * const pinned_biases = (int32_t *)gnaBegin;
    model.CopyBiases(pinned_biases); // puts the biases into the pinned memory
    gnaBegin += buf_size_biases; // fast-forwards current pinned memory pointer to the next free block

    pinned_outputs = (int16_t *)gnaBegin;

    gnaInput = Gna2TensorInit2D(model.NoOfInputs, model.NoOfGroups, Gna2DataTypeInt16, pinned_inputs);
    // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    gnaOutput = Gna2TensorInit2D(model.NoOfOutputs, model.NoOfGroups, Gna2DataTypeInt32, pinned_outputs);
    // parameters needed for the affine transformation are held here
    gnaWeights = Gna2TensorInit2D(model.NoOfOutputs, model.NoOfInputs, Gna2DataTypeInt16, pinned_weights);
    gnaBiases = Gna2TensorInit1D(model.NoOfOutputs, Gna2DataTypeInt32, pinned_biases);
    // no piecewise linear activation function used in this simple example

    const auto status = Gna2ModelCreate(deviceId, &gnaModel, &sampleModelId);
    gnaSelfTest.HandleGnaStatus(status, "Gna2ModelCreate");
}

void SelfTestDevice::BuildSampleRequest()
{
    const auto status = Gna2RequestConfigCreate(sampleModelId, &configId);
    gnaSelfTest.HandleGnaStatus(status, "Gna2RequestConfigCreate");
}

void SelfTestDevice::ConfigRequestBuffer()
{
    auto status = Gna2DeviceGetVersion(deviceId, &deviceVersion);
    gnaSelfTest.HandleGnaStatus(status, "Gna2DeviceGetVersion");

    if(deviceVersion == Gna2DeviceVersionSoftwareEmulation)
    {
        gnaSelfTest.Handle(GSTIT_NO_DEVICE_DETECTED_BY_GNA_LIB);
        logger.Verbose("No device detected by library. Setting Acceleration Mode as Gna2AccelerationModeAuto\n");
        status = Gna2RequestConfigSetAccelerationMode(configId, Gna2AccelerationModeAuto);
        gnaSelfTest.HandleGnaStatus(status, "Gna2RequestConfigSetAccelerationMode");
    }
    else
    {
        logger.Verbose("Setting Acceleration Mode as Gna2AccelerationModeHardware\n");
        status = Gna2RequestConfigSetAccelerationMode(configId, Gna2AccelerationModeHardware);
        gnaSelfTest.HandleGnaStatus(status, "Gna2RequestConfigSetAccelerationMode");
    }
    status = Gna2RequestConfigSetOperandBuffer(configId, 0, 0, pinned_inputs);
    gnaSelfTest.HandleGnaStatus(status, "Adding input buffer to request");
    status = Gna2RequestConfigSetOperandBuffer(configId, 0, 1, pinned_outputs);
    gnaSelfTest.HandleGnaStatus(status, "Adding output buffer to request");
}

void SelfTestDevice::RequestAndWait()
{
    logger.Verbose("Enqueing GNA request for processing\n");
    auto status = Gna2RequestEnqueue(configId, &requestId);
    gnaSelfTest.HandleGnaStatus(status, "GnaRequestEnqueue");
    // Offload effect: other calculations can be done on CPU here, while nnet decoding runs on GNA HW
    status = Gna2RequestWait(requestId, DEFAULT_SELFTEST_TIMEOUT_MS);
    gnaSelfTest.HandleGnaStatus(status, "GnaRequestWait");
}

void SelfTestDevice::CompareResults(const SampleModelForGnaSelfTest& model)
{
    int32_t *outputs = (int32_t *)pinned_outputs;
    int errorCount = 0;
    for (uint32_t j = 0; j < model.NoOfOutputs; j++)
    {
        for (uint32_t i = 0; i < model.NoOfGroups; i++)
        {
            uint32_t idx = j * model.NoOfGroups + i;

            if (outputs[idx] != model.GetRefScore(idx))
            {
                errorCount++;
            }
        }
    }

    if (errorCount != 0)
    {
        GnaSelfTestLogger::Error("FAILED (%d errors in scores)\n", errorCount);
        gnaSelfTest.Handle(GSTIT_ERRORS_IN_SCORES);
    }
    else
    {
        GnaSelfTestLogger::Log("SUCCESSFUL\n");
    }
}

SelfTestDevice::~SelfTestDevice()
{
    // free the pinned memory
    if (deviceOpened)
    {
        for(const auto mem : allocated_mems)
        {
            if (mem != nullptr)
            {
                logger.Verbose("Releasing the memory for GNA, at %p ...\n", mem);
                auto status = Gna2MemoryFree(mem);
                gnaSelfTest.HandleGnaStatus(status, "Gna2MemoryFree");
            }
        }
        logger.Verbose("Closing the device...\n");
        const auto status = Gna2DeviceClose(deviceId);
        gnaSelfTest.HandleGnaStatus(status, "Gna2DeviceClose");
    }
}

void GnaSelfTest::askToContinueOrExit(int exitCode) const
{
    GnaSelfTestLogger::Log("Do you want to go further anyway? [y/N]");
    std::string buf;
    std::getline(std::cin, buf);
    if (buf.empty() || (buf[0] != 'y' && buf[0] != 'Y'))
    {
        GnaSelfTestLogger::Log("Bye\n");
        exit(exitCode);
    }
}

namespace
{
    const std::map<GSTIT, std::string> issueSymbols = {
    { GSTIT_GENERAL_GNA_NO_SUCCESS,                         "GSTIT_GENERAL_GNA_NO_SUCCESS"},
    { GSTIT_DRV_1_INSTEAD_2,                                "GSTIT_DRV_1_INSTEAD_2"},
    { GSTIT_NO_HARDWARE,                                    "GSTIT_NO_HARDWARE"},
    { GSTIT_NUL_DRIVER,                                     "GSTIT_NUL_DRIVER"},
    { GSTIT_DEVICE_OPEN_NO_SUCCESS,                         "GSTIT_DEVICE_OPEN_NO_SUCCESS"},
    { GSTIT_HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY,    "GSTIT_HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY"},
    { GSTIT_GNAALLOC_MEM_ALLOC_FAILED,                      "GSTIT_GNAALLOC_MEM_ALLOC_FAILED"},
    { GSTIT_UNKNOWN_DRIVER,                                 "GSTIT_UNKNOWN_DRIVER"},
    { GSTIT_MALLOC_FAILED,                                  "GSTIT_MALLOC_FAILED"},
    { GSTIT_SETUPDI_ERROR,                                  "GSTIT_SETUPDI_ERROR"},
    { GSTIT_NO_DRIVER,                                      "GSTIT_NO_DRIVER"},
    { GSTIT_ERRORS_IN_SCORES,                               "GSTIT_ERRORS_IN_SCORES"},
    { GSTIT_NO_DEVICE_DETECTED_BY_GNA_LIB,                  "GSTIT_NO_DEVICE_DETECTED_BY_GNA_LIB"} };
}

GnaSelfTestIssue::GnaSelfTestIssue(GSTIT type) :issueType{ type }
{
    //TODO: add more details on the exact place in the Bring up guide
    symbol = issueSymbols.at(type);
}
std::string GnaSelfTestIssue::GetDescription() const
{
    std::string desc = "ERROR Issue " + symbol + ": Find additional instructions in Bring up guide\n";
    return desc;
}

void GnaSelfTest::Handle(const GnaSelfTestIssue& issue) const
{
    GnaSelfTestLogger::Error("%s", issue.GetDescription().c_str());

    if (config.ContinueMode())
    {
        if (config.PauseMode()) {
            askToContinueOrExit(issue.ExitCode());
        }
        GnaSelfTestLogger::Log("WARNING Continuing the execution after the problem\n");
    }
    else
    {
        logger.Verbose("You can run \"gna-self-test -c\" to continue gna-self-test despite the problem\n");
        exit(issue.ExitCode());
    }
}
