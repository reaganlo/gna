/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include <cstdio>
#include <cstdlib>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#endif

// Enable safe functions compatibility
#if defined(__STDC_SECURE_LIB__)
#define __STDC_WANT_SECURE_LIB__ 1
#elif defined(__STDC_LIB_EXT1__)
#define STDC_WANT_LIB_EXT1 1
#else
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) memcpy(_Destination, _Source, _SourceSize)
#endif

#include <cstring>
#include <fstream>

#include "gna-api-dumper.h"

void print_outputs(
    int32_t *outputs,
    uint32_t nRows,
    uint32_t nColumns
)
{
    printf("\nOutputs:\n");
    for(uint32_t i = 0; i < nRows; ++i)
    {
        for(uint32_t j = 0; j < nColumns; ++j)
        {
            printf("%d\t", outputs[i*nColumns + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

void* customAlloc(uint32_t dumpedModelSize)
{
    if (0 == dumpedModelSize)
    {
        printf("customAlloc has invalid dump model size: %zu\n", dumpedModelSize);
        exit(-GNA_INVALIDMEMSIZE);
    }
    return _mm_malloc(dumpedModelSize, 4096);
}

int main(int argc, char *argv[])
{
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
    /* Dump binary inputs (for reference only) */
    std::ofstream inputStream("inputs.bin", std::ios::out | std::ios::binary);
    inputStream.write(reinterpret_cast<const char*>(inputs), (16 * 4) * sizeof(int16_t));

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

    intel_gna_status_t status = GNA_SUCCESS; // for simplicity sake status codes are not examined after api functions calls
                                             // it is highly recommended to inspect the status every time, and act accordingly
                                             // open the device
   gna_device_id deviceNumber;
     status = GnaDeviceGetCount(&deviceNumber);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaDeviceGetCount failed: %s\n", GnaStatusToString(status));
        exit(-status);
    }
    gna_device_id deviceIndex = 0;
    // open the device
    status = GnaDeviceOpen(deviceIndex);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaDeviceOpen failed: %s\n", GnaStatusToString(status));
        exit(-status);
    }

    /* Calculate model memory parameters for GnaAlloc. */
    int buf_size_weights     = ALIGN64(sizeof(weights)); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs      = ALIGN64(sizeof(inputs));
    int buf_size_biases      = ALIGN64(sizeof(biases));
    int buf_size_outputs     = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)
    int buf_size_tmp_outputs = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)

    uint32_t rw_buffer_size = ALIGN(buf_size_inputs + buf_size_outputs + buf_size_tmp_outputs, 0x1000);
    uint32_t bytes_requested = rw_buffer_size + buf_size_weights + buf_size_biases;

    // call GnaAlloc (obtains pinned memory shared with the device)
    const uint32_t layer_count = 1;
    uint32_t bytes_granted;

    void *memory;
    status = GnaAlloc(bytes_requested, &bytes_granted, &memory);
    if (GNA_SUCCESS != status || nullptr == memory)
    {
        printf("GnaAlloc failed: %s\n", GnaStatusToString(status));
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }

    /* Prepare model memory layout. */
    uint8_t *model_memory = (uint8_t*)memory;

    // RW region.
    uint8_t *rw_buffers = model_memory;

    int16_t *pinned_inputs = (int16_t*)rw_buffers;
    memcpy_s(pinned_inputs, buf_size_inputs, inputs, sizeof(inputs));      // puts the inputs into the pinned memory
    rw_buffers += buf_size_inputs;               // fast-forwards current pinned memory pointer to the next free block

    int16_t *pinned_outputs = (int16_t*)rw_buffers;
    rw_buffers += buf_size_outputs;              // fast-forwards the current pinned memory pointer by the space needed for outputs

    int32_t *tmp_outputs_buffer = (int32_t*)rw_buffers;// the last free block will be used for GNA's scratch pad

    // RO region

    model_memory += rw_buffer_size;
    int16_t *weights_buffer = (int16_t*)model_memory;
    memcpy_s(weights_buffer, buf_size_weights, weights, sizeof(weights));   // puts the weights into the pinned memory
    model_memory += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    int32_t *biases_buffer = (int32_t*)model_memory;
    memcpy_s(biases_buffer, buf_size_biases, biases, sizeof(biases));      // puts the biases into the pinned memory
    model_memory += buf_size_biases;                  // fast-forwards current pinned memory pointer to the next free block

    /* Prepare neural network topology. */

    intel_nnet_type_t nnet;  // main neural network container
    nnet.nGroup = 4;         // grouping factor (1-8), specifies how many input vectors are simultaneously run through the nnet
    nnet.nLayers = 1;        // number of hidden layers, using 1 for simplicity sake
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));   // container for layer definitions
    if (nullptr == nnet.pLayers)
    {
        printf("Allocation for nnet.pLayers failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }

    intel_affine_func_t affine_func;       // parameters needed for the affine transformation are held here
    affine_func.nBytesPerWeight = GNA_INT16;
    affine_func.nBytesPerBias = GNA_INT32;
    affine_func.pWeights = weights_buffer;
    affine_func.pBiases = biases_buffer;

    intel_pwl_func_t pwl;                  // no piecewise linear activation function used in this simple example
    pwl.nSegments = 0;
    pwl.pSegments = NULL;

    intel_affine_layer_t affine_layer;     // affine layer combines the affine transformation and activation function
    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    intel_nnet_layer_t nnet_layer = nnet.pLayers[0];         // contains the definition of a single layer
    nnet_layer.nInputColumns = nnet.nGroup;
    nnet_layer.nInputRows = 16;
    nnet_layer.nOutputColumns = nnet.nGroup;
    nnet_layer.nOutputRows = 8;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT32;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32; // this is always 4 bytes
    nnet_layer.mode = INTEL_INPUT_OUTPUT;
    nnet_layer.operation = INTEL_AFFINE;
    nnet_layer.pLayerStruct = &affine_layer;

    nnet_layer.pInputs = pinned_inputs;
    nnet_layer.pOutputsIntermediate = tmp_outputs_buffer;
    nnet_layer.pOutputs = pinned_outputs;

    // puts the layer into the main network container
    // if there was another layer to add, it would get copied to nnet.pLayers + 1
    memcpy_s(nnet.pLayers, nnet.nLayers * sizeof(intel_nnet_layer_t), &nnet_layer, sizeof(nnet_layer));

    gna_model_id model_id;
    status = GnaModelCreate(deviceIndex, &nnet, &model_id);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaModelCreate failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }

    intel_gna_model_header model_header;
    void* dumped_model = GnaModelDump(model_id, GNA_1_0_EMBEDDED, &model_header, &status, customAlloc);
    if (GNA_SUCCESS != status || NULL == dumped_model)
    {
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }

    model_header.rw_region_size = rw_buffer_size;

    /* Save dumped model with header to stream or file. */
    std::ofstream dumpStream("model.bin", std::ios::out | std::ios::binary);
    dumpStream.write(reinterpret_cast<const char*>(&model_header), sizeof(intel_gna_model_header));
    dumpStream.write(reinterpret_cast<const char*>(dumped_model), model_header.model_size);

    /* Release dump memory if no longer needed. */
    _mm_free(dumped_model);

    gna_request_cfg_id config_id;
    status = GnaRequestConfigCreate(model_id, &config_id);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigCreate failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }
    status = GnaRequestConfigBufferAdd(config_id, InputComponent, 0, pinned_inputs);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigBufferAdd InputComponent failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }
    status = GnaRequestConfigBufferAdd(config_id, OutputComponent, 0, pinned_outputs);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigBufferAdd OutputComponent failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }
    status = GnaRequestConfigEnableHardwareConsistency(config_id, GNA_SUE_CREEK);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigEnableHardwareConsistency for GNA_SUE_CREEK failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }
    status = GnaRequestConfigEnforceAcceleration(config_id, GNA_GENERIC);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigEnforceAcceleration GNA_GENERIC failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }

    // calculate on GNA HW (non-blocking call)
    gna_request_id request_id;     // this gets filled with the actual id later on
    status = GnaRequestEnqueue(config_id, &request_id);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestEnqueue failed: %s\n", GnaStatusToString(status));
        GnaFree(memory);
        GnaDeviceClose(deviceIndex);
        exit(-status);
    }

    /**************************************************************************************************
     * Offload effect: other calculations can be done on CPU here, while nnet decoding runs on GNA HW *
     **************************************************************************************************/

    // wait for HW calculations (blocks until the results are ready)
    gna_timeout timeout = 1000;
    status = GnaRequestWait(request_id, timeout);     // after this call, outputs can be inspected under nnet.pLayers->pOutputs

    print_outputs((int32_t*)pinned_outputs, nnet.pLayers->nOutputRows, nnet.pLayers->nOutputColumns);

    /* Dump binary outputs (for reference only) */
    std::ofstream outputsStream("outputs.bin", std::ios::out | std::ios::binary);
    outputsStream.write(reinterpret_cast<const char*>(nnet.pLayers[nnet.nLayers - 1].pOutputs), buf_size_outputs);

                                                      // -177  -85   29   28
    // free the pinned memory                         //   96 -173   25  252
    status = GnaFree(memory);                         // -160  274  157  -29
                                                      //   48  -60  158  -29
    // free heap allocations                          //   26   -2  -44 -251
    free(nnet.pLayers);                               // -173  -70   -1 -323
                                                      //   99  144   38  -63
    // close the device                               //   20   56 -103   10
    status = GnaDeviceClose(deviceIndex);

    return 0;
}
