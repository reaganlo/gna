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

void HandleGnaStatus(const intel_gna_status_t& status, const char * what = "") {
    LOG("%s: %s\n", what, GnaStatusToString(status));
    if (status != GNA_SUCCESS) {
        GnaSelfTestIssue::GENERAL_GNA_NO_SUCCESS.Handle();
    }
}

SelfTestDevice::SelfTestDevice()
{
    // open the device
    intel_gna_status_t status = GnaDeviceOpen(1, &deviceId);

    const char * statusStr = GnaStatusToString(status);
    LOG("SelfTestDevice: %s\n", statusStr);
    if (deviceId == GNA_DEVICE_INVALID || status != GNA_SUCCESS)
    {
        LOG("selfTestOpenDevice FAILED\n");
        GnaSelfTestIssue::DEVICE_OPEN_NO_SUCCESS.Handle();
    }
}

// obtains pinned memory shared with the device
void SelfTestDevice::Alloc(const uint32_t bytesRequested, const uint16_t layerCount, const uint16_t gmmCount)
{

    //uint8_t *pinned_mem_ptr = (uint8_t*)GNAAlloc(gna_handle, bytes_requested, &bytes_granted);
    uint32_t sizeGranted;
    pinned_mem_ptr = reinterpret_cast<uint8_t*>(GnaAlloc(deviceId, bytesRequested, layerCount, gmmCount, &sizeGranted));

    LOG("Requested: %i Granted: %i\n", (int)bytesRequested, (int)sizeGranted);

    if (nullptr == pinned_mem_ptr)
    {
        LOG("GnaAlloc: Memory allocation FAILED\n");
        GnaSelfTestIssue::GNAALLOC_MEM_ALLOC_FAILED.Handle();
    }
}

//TODO refactor into factory
void SelfTestDevice::SampleModelCreate()
{
    nnet.nGroup = 4;         // grouping factor (1-8), specifies how many input vectors are simultaneously run through the nnet
    nnet.nLayers = 1;        // number of hidden layers, using 1 for simplicity sake
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));   // container for layer definitions
    if (nullptr == nnet.pLayers)
    {
        LOG("calloc: Allocation for nnet.pLayers FAILED.\n");
        GnaSelfTestIssue::MALLOC_FAILED.Handle();
    }

    int16_t weights[8 * 16] = {                                          // sample weight matrix (8 rows, 16 cols)
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,  // in case of affine layer this is the left operand of matrix mul
        -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,  // in this sample the numbers are random and meaningless
        2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
        0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
        -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
        -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
        0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
        2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
    };

    int16_t inputs[16 * 4] = {      // sample input matrix (16 rows, 4 cols), consists of 4 input vectors (grouping of 4 is used)
        -5,  9, -7,  4,             // in case of affine layer this is the right operand of matrix mul
        5, -4, -7,  4,             // in this sample the numbers are random and meaningless
        0,  7,  1, -7,
        1,  6,  7,  9,
        2, -4,  9,  8,
        -5, -1,  2,  9,
        -8, -8,  8,  1,
        -7,  2, -1, -1,
        -9, -5, -8,  5,
        0, -1,  3,  9,
        0,  8,  1, -2,
        -9,  8,  0, -7,
        -9, -8, -1, -4,
        -3, -7, -2,  3,
        -8,  0,  1,  3,
        -4, -6, -8, -2
    };

    int32_t biases[8] = {      // sample bias vector, will get added to each of the four output vectors
        5,                    // in this sample the numbers are random and meaningless
        4,
        -2,
        5,
        -7,
        -5,
        4,
        -1
    };

    int buf_size_weights = ALIGN64(sizeof(weights)); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs = ALIGN64(sizeof(inputs));
    int buf_size_biases = ALIGN64(sizeof(biases));
    int buf_size_outputs = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)
    int buf_size_tmp_outputs = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)

                                                         // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    //uint32_t bytes_granted;

    Alloc(bytes_requested, nnet.nLayers, 0);

    int16_t *pinned_weights = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_weights, weights, sizeof(weights));   // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    pinned_inputs = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_inputs, inputs, sizeof(inputs));      // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;                  // fast-forwards current pinned memory pointer to the next free block

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, biases, sizeof(biases));      // puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases;                  // fast-forwards current pinned memory pointer to the next free block

    pinned_outputs = (int16_t*)pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;                 // fast-forwards the current pinned memory pointer by the space needed for outputs

    int32_t *pinned_tmp_outputs = (int32_t*)pinned_mem_ptr;      // the last free block will be used for GNA's scratch pad

    intel_affine_func_t affine_func;       // parameters needed for the affine transformation are held here
    affine_func.nBytesPerWeight = 2;
    affine_func.nBytesPerBias = 4;
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    intel_pwl_func_t pwl;                  // no piecewise linear activation function used in this simple example
    pwl.nSegments = 0;
    pwl.pSegments = nullptr;

    intel_affine_layer_t affine_layer;     // affine layer combines the affine transformation and activation function
    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    intel_nnet_layer_t* nnet_layer = &nnet.pLayers[0];   // contains the definition of a single layer
    nnet_layer->nInputColumns = nnet.nGroup;
    nnet_layer->nInputRows = 16;
    nnet_layer->nOutputColumns = nnet.nGroup;
    nnet_layer->nOutputRows = 8;
    nnet_layer->nBytesPerInput = 2;
    nnet_layer->nBytesPerOutput = 4;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet_layer->nBytesPerIntermediateOutput = 4; // this is always 4 bytes
    nnet_layer->nLayerKind = INTEL_AFFINE;
    nnet_layer->pLayerStruct = &affine_layer;
    nnet_layer->pInputs = pinned_inputs;
    nnet_layer->pOutputsIntermediate = pinned_tmp_outputs;
    nnet_layer->pOutputs = pinned_outputs;
    nnet_layer->type = intel_layer_type_t::INTEL_INPUT_OUTPUT;
    intel_gna_status_t status = GnaModelCreate(deviceId, &nnet, &sampleModelId);
    HandleGnaStatus(status, "GnaModelCreate");
}

void SelfTestDevice::BuildSampleRequest()
{
    auto status = GnaModelRequestConfigAdd(sampleModelId, &configId);
    HandleGnaStatus(status, "GnaModelRequestConfigAdd");
}

void SelfTestDevice::ConfigRequestBuffer()
{
    auto status = GnaRequestConfigBufferAdd(configId, gna_buffer_type::GNA_IN, 0, pinned_inputs);
    HandleGnaStatus(status, "Adding input buffer to request");
    status = GnaRequestConfigBufferAdd(configId, gna_buffer_type::GNA_OUT, 0, pinned_outputs);
    HandleGnaStatus(status, "Adding output buffer to request");
}

void SelfTestDevice::RequestAndWait()
{
    LOG("Enqueing GNA request for processing\n")
    auto status = GnaRequestEnqueue(configId, gna_acceleration::GNA_AUTO, &requestId);
    // Offload effect: other calculations can be done on CPU here, while nnet decoding runs on GNA HW
    status = GnaRequestWait(requestId, DEFAULT_SELFTEST_TIMEOUT_MS);
    HandleGnaStatus(status, "GnaRequestEnqueue");
    HandleGnaStatus(status, "GnaRequestWait");
}

void SelfTestDevice::CompareResults()
{
    int32_t refScores[] = {
        -177,  -85,  29,   28,
        96, -173,  25,  252,
        -160,  274, 157,  -29,
        48,  -60, 158,  -29,
        26,   -2, -44, -251,
        -173,  -70,  -1, -323,
        99,  144,  38,  -63,
        20,   56,-103,   10
    };

    int32_t *outputs = (int32_t*)pinned_outputs;
    int errorCount = 0;
    for (uint32_t j = 0; j < nnet.pLayers[0].nOutputRows; j++)
    {
        for (uint32_t i = 0; i < nnet.nGroup; i++)
        {
            uint32_t idx = j * nnet.nGroup + i;

            if (outputs[idx] != refScores[idx])
                errorCount++;
        }
    }

    if (errorCount != 0)
        LOG("FAILED (%d errors in scores)\n", errorCount)
    else
        LOG("SUCCESSFULL\n");

}

SelfTestDevice::~SelfTestDevice() {
    // free the pinned memory
    if (deviceId != GNA_DEVICE_INVALID) {
        if (pinned_mem_ptr != nullptr) {
            LOG("releasing the gna allocated memory...\n");
            auto status = GnaFree(deviceId);
            HandleGnaStatus(status, "GnaFree");
            pinned_mem_ptr = nullptr;
        }
        LOG("releasing the device...\n");
        auto status = GnaDeviceClose(deviceId);
        HandleGnaStatus(status, "GnaDeviceClose");
    }

    // free heap allocations
    if (nnet.pLayers != nullptr) {
        LOG("releasing the network memory\n");
        free(nnet.pLayers);
    }
}

void PressEnterToContinue()
{
    LOG("----------------------------------------------\n");
    //LOG("Press [Enter] to continue");
    //std::string buf;
    //std::getline(std::cin, buf);
}

//TODO: add more details on the exact place in the Bring up guide
#define DEF_GNASELFTESTISSUE(NAME,INFO) const GnaSelfTestIssue GnaSelfTestIssue::NAME = GnaSelfTestIssue("ERROR Issue " INFO ": Find additional instructions in Bring up guide\n");
DEF_GNASELFTESTISSUE(GENERAL_GNA_NO_SUCCESS,"GENERAL_GNA_NO_SUCCESS");
DEF_GNASELFTESTISSUE(DRV_1_INSTEAD_2,"DRV_1_INSTEAD_2");
DEF_GNASELFTESTISSUE(NO_HARDWARE,"NO_HARDWARE");
DEF_GNASELFTESTISSUE(NUL_DRIVER,"NUL_DRIVER");
DEF_GNASELFTESTISSUE(DEVICE_OPEN_NO_SUCCESS,"DEVICE_OPEN_NO_SUCCESS");
DEF_GNASELFTESTISSUE(HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY,"HARDWARE_OR_DRIVER_NOT_INITIALIZED_PROPERLY");
DEF_GNASELFTESTISSUE(GNAALLOC_MEM_ALLOC_FAILED,"GNAALLOC_MEM_ALLOC_FAILED");
DEF_GNASELFTESTISSUE(UNKNOWN_DRIVER,"UNKNOWN_DRIVER");
DEF_GNASELFTESTISSUE(MALLOC_FAILED,"MALLOC_FAILED");
DEF_GNASELFTESTISSUE(SETUPDI_ERROR,"SETUPDI_ERROR");
DEF_GNASELFTESTISSUE(NO_DRIVER,"NO_DRIVER");

void GnaSelfTestIssue::Handle() const
{
    LOG("%s",info);
    LOG("Do you want to go further anyway? [y/N]");
    std::string buf;
    std::getline(std::cin, buf);
    if (buf.size() == 0 || (buf[0] != 'y'&&buf[0] != 'Y')) {
        LOG("Bye\n");
        exit(0);
    }
}
