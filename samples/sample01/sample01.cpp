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

// Enable safe functions compatibility
#if defined(__STDC_SECURE_LIB__)
#define __STDC_WANT_SECURE_LIB__ 1
#elif defined(__STDC_LIB_EXT1__)
#define STDC_WANT_LIB_EXT1 1
#else
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) memcpy(_Destination, _Source, _SourceSize)
#endif

#include <cstring>

#include "gna-api.h"

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

void print_outputs16(
    int16_t *outputs,
    uint32_t nRows,
    uint32_t nColumns
)
{
    printf("\nOutputs:\n");
    for(uint32_t i = 0; i < nRows; ++i)
    {
        for(uint32_t j = 0; j < nColumns; ++j)
        {
            printf("%hd\t", outputs[i*nColumns + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

int main(int argc, char *argv[])
{
    gna_status_t status = GNA_SUCCESS;

    // open the device
    gna_device_id gna_handle;
    status = GnaDeviceOpen(1, &gna_handle);
    if (GNA_SUCCESS!= status)
    {
        printf("GNADeviceOpen failed: %s\n", GnaStatusToString(status));
        exit(-status);
    }

    intel_nnet_type_t nnet;  // main neural network container
    nnet.nGroup = 4;         // grouping factor (1-8), specifies how many input vectors are simultaneously run through the nnet
    nnet.nLayers = 1;        // number of hidden layers, using 1 for simplicity sake
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));   // container for layer definitions
    if (nullptr == nnet.pLayers)
    {
        printf("Allocation for nnet.pLayers failed.\n");
        GnaDeviceClose(gna_handle);
        exit(-1);
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

    intel_pwl_segment_t pwl_segs[]
    {
        {-512, -51, 256},
        {0, 0, 256}
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

    int buf_size_weights     = ALIGN64(sizeof(weights)); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs      = ALIGN64(sizeof(inputs));
    int buf_size_biases      = ALIGN64(sizeof(biases));
    int buf_size_pwl         = ALIGN64(sizeof(pwl_segs));
    int buf_size_outputs     = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)
    int buf_size_tmp_outputs = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)

    // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_pwl + buf_size_outputs + buf_size_tmp_outputs;
    uint32_t bytes_granted;

    // call GNAAlloc (obtains pinned memory shared with the device)
    uint8_t *pinned_mem_ptr = (uint8_t*)GnaAlloc(gna_handle, bytes_requested, 1, 0, &bytes_granted);
    if (nullptr == pinned_mem_ptr)
    {
        printf("GnaAlloc failed.\n");
        GnaDeviceClose(gna_handle);
        exit(-1);
    }

    int16_t *pinned_weights = (int16_t*)pinned_mem_ptr;
    memcpy_s(pinned_weights, buf_size_weights, weights, sizeof(weights));   // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    int16_t *pinned_inputs = (int16_t*)pinned_mem_ptr;
    memcpy_s(pinned_inputs, buf_size_inputs, inputs, sizeof(inputs));      // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;                  // fast-forwards current pinned memory pointer to the next free block

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy_s(pinned_biases, buf_size_biases, biases, sizeof(biases));      // puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases;                  // fast-forwards current pinned memory pointer to the next free block


    intel_pwl_segment_t *pinned_pwl = (intel_pwl_segment_t*)pinned_mem_ptr;
    memcpy_s(pinned_pwl, buf_size_pwl, pwl_segs, sizeof(pwl_segs));      // puts the pwl into the pinned memory
    pinned_mem_ptr += buf_size_pwl;                  // fast-forwards current pinned memory pointer to the next free block

    int16_t *pinned_outputs = (int16_t*)pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;                 // fast-forwards the current pinned memory pointer by the space needed for outputs

    int32_t *pinned_tmp_outputs = (int32_t*)pinned_mem_ptr;      // the last free block will be used for GNA's scratch pad

    intel_affine_func_t affine_func;       // parameters needed for the affine transformation are held here
    affine_func.nBytesPerWeight = GNA_INT16;
    affine_func.nBytesPerBias = GNA_INT32;
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    intel_pwl_func_t pwl;                  // no piecewise linear activation function used in this simple example
    pwl.nSegments = 2;
    pwl.pSegments = pinned_pwl;

    intel_affine_layer_t affine_layer;     // affine layer combines the affine transformation and activation function
    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    intel_nnet_layer_t nnet_layer;         // contains the definition of a single layer
    nnet_layer.nInputColumns = nnet.nGroup;
    nnet_layer.nInputRows = 16;
    nnet_layer.nOutputColumns = nnet.nGroup;
    nnet_layer.nOutputRows = 8;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT16;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32; // this is always 4 bytes
    nnet_layer.mode = INTEL_INPUT_OUTPUT;
    nnet_layer.operation = INTEL_AFFINE;
    nnet_layer.pLayerStruct = &affine_layer;
    nnet_layer.pInputs = nullptr;
    nnet_layer.pOutputsIntermediate = pinned_tmp_outputs;
    nnet_layer.pOutputs = nullptr;

    memcpy_s(nnet.pLayers, sizeof(intel_nnet_layer_t), &nnet_layer, sizeof(nnet_layer));   // puts the layer into the main network container
                                                             // if there was another layer to add, it would get copied to nnet.pLayers + 1

    gna_model_id model_id;
    status = GnaModelCreate(gna_handle, &nnet, &model_id);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaModelCreate failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }

    gna_request_cfg_id config_id;
    status = GnaRequestConfigCreate(model_id, &config_id);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigCreate failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }
    status = GnaRequestConfigBufferAdd(config_id, InputComponent, 0, pinned_inputs);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigBufferAdd InputComponent failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }
    status = GnaRequestConfigBufferAdd(config_id, OutputComponent, 0, pinned_outputs);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigBufferAdd OutputComponent failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }
    status = GnaRequestConfigEnforceAcceleration(config_id, GNA_GENERIC);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestConfigEnforceAcceleration GNA_GENERIC failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }

    // calculate on GNA HW (non-blocking call)
    gna_request_id request_id;     // this gets filled with the actual id later on
    status = GnaRequestEnqueue(config_id, &request_id);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestEnqueue failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }
    /**************************************************************************************************
     * Offload effect: other calculations can be done on CPU here, while nnet decoding runs on GNA HW *
     **************************************************************************************************/

    // wait for HW calculations (blocks until the results are ready)
    gna_timeout timeout = 1000;
    status = GnaRequestWait(request_id, timeout);     // after this call, outputs can be inspected under nnet.pLayers->pOutputs
    if (GNA_SUCCESS!= status)
    {
        printf("GnaRequestWait failed: %s\n", GnaStatusToString(status));
        GnaFree(gna_handle);
        GnaDeviceClose(gna_handle);
        exit(-status);
    }

    print_outputs16((int16_t*)pinned_outputs, nnet.pLayers->nOutputRows, nnet.pLayers->nOutputColumns);

    // free the pinned memory
    status = GnaFree(gna_handle);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaFree failed: %s\n", GnaStatusToString(status));
        GnaDeviceClose(gna_handle);
        exit(-status);
    }
    // Results:
    // -177  -85   29   28
    //   96 -173   25  252
    // -160  274  157  -29
    //   48  -60  158  -29
    //   26   -2  -44 -251
    // -173  -70   -1 -323
    //   99  144   38  -63
    //   20   56 -103   10

    // free heap allocations
    free(nnet.pLayers);

    // close the device
    status = GnaDeviceClose(gna_handle);
    if (GNA_SUCCESS!= status)
    {
        printf("GnaDeviceClose failed: %s\n", GnaStatusToString(status));
        exit(-status);
    }

    return 0;
}
