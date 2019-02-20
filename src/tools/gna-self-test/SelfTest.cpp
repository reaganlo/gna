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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <map>

void GnaSelfTest::HandleGnaStatus(const intel_gna_status_t &status, const char *what) const
{
    logger.Verbose("%s: %s\n", what, GnaStatusToString(status));
    if (status != GNA_SUCCESS)
    {
        Handle(GSTIT_GENERAL_GNA_NO_SUCCESS);
    }
}

SelfTestDevice::SelfTestDevice(const GnaSelfTest& gst) : gnaSelfTest{ gst }
{
    // open the device
    intel_gna_status_t status = GnaDeviceOpen(deviceId);

    const char *statusStr = GnaStatusToString(status);
    logger.Verbose("GnaDeviceOpen: %s\n", statusStr);
    if (status != GNA_SUCCESS)
    {
        logger.Error("GnaDeviceOpen FAILED\n");
        gnaSelfTest.Handle(GSTIT_DEVICE_OPEN_NO_SUCCESS);
    }
    else
    {
        deviceOpened = true;
    }
}

// obtains pinned memory shared with the device
void SelfTestDevice::Alloc(const uint32_t bytesRequested, const uint16_t layerCount, const uint16_t gmmCount)
{
    uint32_t sizeGranted;
    pinned_mem_ptr = reinterpret_cast<uint8_t *>(GnaAlloc(deviceId, bytesRequested, layerCount, gmmCount, &sizeGranted));

    logger.Verbose("Requested: %i Granted: %i\n", (int)bytesRequested, (int)sizeGranted);

    if (nullptr == pinned_mem_ptr)
    {
        logger.Error("GnaAlloc: Memory allocation FAILED\n");
        gnaSelfTest.Handle(GSTIT_GNAALLOC_MEM_ALLOC_FAILED);
    }
}

void SelfTestDevice::SampleModelCreate(const SampleModelForGnaSelfTest& model)
{
    nnet.nGroup = 4;  // grouping factor (1-8), specifies how many input vectors are simultaneously run through the nnet
    nnet.nLayers = 1; // number of hidden layers, using 1 for simplicity sake

    nnet.pLayers = (intel_nnet_layer_t *)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t)); // container for layer definitions
    if (nullptr == nnet.pLayers)
    {
        logger.Error("calloc: Allocation for nnet.pLayers FAILED.\n");
        gnaSelfTest.Handle(GSTIT_MALLOC_FAILED);
    }

    int buf_size_weights = ALIGN64(model.GetWeightsByteSize()); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs = ALIGN64(model.GetInputsByteSize());
    int buf_size_biases = ALIGN64(model.GetBiasesByteSize());
    int buf_size_outputs = ALIGN64(model.GetRefScoresByteSize());
    int buf_size_tmp_outputs = ALIGN64(model.GetRefScoresByteSize());

    // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    //uint32_t bytes_granted;

    Alloc(bytes_requested, static_cast<uint16_t>(nnet.nLayers), static_cast<uint16_t>(0));

    int16_t *pinned_weights = (int16_t *)pinned_mem_ptr;
    model.CopyWeights(pinned_weights);    // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;   // fast-forwards current pinned memory pointer to the next free block

    pinned_inputs = (int16_t *)pinned_mem_ptr;
    model.CopyInputs(pinned_inputs);    // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;  // fast-forwards current pinned memory pointer to the next free block

    int32_t *pinned_biases = (int32_t *)pinned_mem_ptr;
    model.CopyBiases(pinned_biases); // puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases; // fast-forwards current pinned memory pointer to the next free block

    pinned_outputs = (int16_t *)pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs; // fast-forwards the current pinned memory pointer by the space needed for outputs

    int32_t *pinned_tmp_outputs = (int32_t *)pinned_mem_ptr; // the last free block will be used for GNA's scratch pad

    intel_affine_func_t affine_func; // parameters needed for the affine transformation are held here
    affine_func.nBytesPerWeight = 2;
    affine_func.nBytesPerBias = 4;
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    intel_pwl_func_t pwl; // no piecewise linear activation function used in this simple example
    pwl.nSegments = 0;
    pwl.pSegments = nullptr;

    intel_affine_layer_t affine_layer; // affine layer combines the affine transformation and activation function
    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    intel_nnet_layer_t *nnet_layer = &nnet.pLayers[0]; // contains the definition of a single layer
    nnet_layer->nInputColumns = nnet.nGroup;
    nnet_layer->nInputRows = 16;
    nnet_layer->nOutputColumns = nnet.nGroup;
    nnet_layer->nOutputRows = 8;
    nnet_layer->nBytesPerInput = 2;
    nnet_layer->nBytesPerOutput = 4;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet_layer->nBytesPerIntermediateOutput = 4; // this is always 4 bytes
    nnet_layer->operation = INTEL_AFFINE;
    nnet_layer->pLayerStruct = &affine_layer;
    nnet_layer->pInputs = pinned_inputs;
    nnet_layer->pOutputsIntermediate = pinned_tmp_outputs;
    nnet_layer->pOutputs = pinned_outputs;
    nnet_layer->mode = INTEL_INPUT_OUTPUT;
    intel_gna_status_t status = GnaModelCreate(deviceId, &nnet, &sampleModelId);
    gnaSelfTest.HandleGnaStatus(status, "GnaModelCreate");
}

void SelfTestDevice::BuildSampleRequest()
{
    auto status = GnaRequestConfigCreate(sampleModelId, &configId);
    gnaSelfTest.HandleGnaStatus(status, "GnaRequestConfigCreate");
}

void SelfTestDevice::ConfigRequestBuffer()
{
    auto status = GnaRequestConfigBufferAdd(configId, GnaComponentType::InputComponent, 0, pinned_inputs);
    gnaSelfTest.HandleGnaStatus(status, "Adding input buffer to request");
    status = GnaRequestConfigBufferAdd(configId, GnaComponentType::OutputComponent, 0, pinned_outputs);
    gnaSelfTest.HandleGnaStatus(status, "Adding output buffer to request");
}

void SelfTestDevice::RequestAndWait()
{
    logger.Verbose("Enqueing GNA request for processing\n");
    auto status = GnaRequestEnqueue(configId, &requestId);
    // Offload effect: other calculations can be done on CPU here, while nnet decoding runs on GNA HW
    status = GnaRequestWait(requestId, DEFAULT_SELFTEST_TIMEOUT_MS);
    gnaSelfTest.HandleGnaStatus(status, "GnaRequestEnqueue");
    gnaSelfTest.HandleGnaStatus(status, "GnaRequestWait");
}

void SelfTestDevice::CompareResults(const SampleModelForGnaSelfTest& model)
{
    int32_t *outputs = (int32_t *)pinned_outputs;
    int errorCount = 0;
    for (uint32_t j = 0; j < nnet.pLayers[0].nOutputRows; j++)
    {
        for (uint32_t i = 0; i < nnet.nGroup; i++)
        {
            uint32_t idx = j * nnet.nGroup + i;

            if (outputs[idx] != model.GetRefScore(idx))
                errorCount++;
        }
    }

    if (errorCount != 0)
    {
        logger.Error("FAILED (%d errors in scores)\n", errorCount);
        gnaSelfTest.Handle(GSTIT_ERRORS_IN_SCORES);
    }
    else
        logger.Log("SUCCESSFULL\n");
}

SelfTestDevice::~SelfTestDevice()
{
    // free the pinned memory
    if (deviceOpened)
    {
        if (pinned_mem_ptr != nullptr)
        {
            logger.Verbose("releasing the gna allocated memory...\n");
            auto status = GnaFree(deviceId);
            gnaSelfTest.HandleGnaStatus(status, "GnaFree");
            pinned_mem_ptr = nullptr;
        }
        logger.Verbose("releasing the device...\n");
        auto status = GnaDeviceClose(deviceId);
        gnaSelfTest.HandleGnaStatus(status, "GnaDeviceClose");
    }

    // free heap allocations
    if (nnet.pLayers != nullptr)
    {
        logger.Verbose("releasing the network memory\n");
        free(nnet.pLayers);
    }
}

void GnaSelfTest::askToContinueOrExit(int exitCode) const
{
    logger.Log("Do you want to go further anyway? [y/N]");
    std::string buf;
    std::getline(std::cin, buf);
    if (buf.size() == 0 || (buf[0] != 'y' && buf[0] != 'Y'))
    {
        logger.Log("Bye\n");
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
    { GSTIT_ERRORS_IN_SCORES,                               "GSTIT_ERRORS_IN_SCORES"} };
};
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
    logger.Error("%s", issue.GetDescription().c_str());

    if (config.ContinueMode())
    {
        if (config.PauseMode()) {
            askToContinueOrExit(issue.ExitCode());
        }
        logger.Log("WARNING Continuing the execution after the problem\n");
    }
    else
    {
        logger.Verbose("You can run \"gna-self-test -c\" to continue gna-self-test despite the problem\n");
        exit(issue.ExitCode());
    }
}
