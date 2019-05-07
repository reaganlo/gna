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

// Enable safe functions compatibility
#define __STDC_WANT_SECURE_LIB__ 1

#include "common.h"
#include "gna-api.h"
#include "gna-api-dumper.h"
#include "gna2-model-export-impl.h"

#include <gtest/gtest.h>
#include <cstring>

#ifndef __STDC_LIB_EXT1__
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) memcpy(_Destination, _Source, _SourceSize)
#endif

class TestSimpleModel : public testing::Test
{
protected:
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

    const int buf_size_weights = ALIGN64(sizeof(weights));
    const int buf_size_inputs = ALIGN64(sizeof(inputs));
    const int buf_size_biases = ALIGN64(sizeof(biases));
    const int buf_size_outputs = ALIGN64(8 * 4 * 4);
    const int buf_size_tmp_outputs = ALIGN64(8 * 4 * 4);

    static void * Allocator(uint32_t size)
    {
        return _mm_malloc(size, 4096);
    }

    static void Free(void * mem)
    {
        return _mm_free(mem);
    }

    const uint32_t expectedModelSize = 4544;
    const uint32_t expectedHeaderSize = 64;

    const uint32_t expected_headerHash = 0x50e2e119;
    const uint32_t expected_modelHash = 0xe99d8e89;
    const uint32_t expected_fileHash = 0xacd78845;

    // Based on
    // Simple public domain implementation of the standard CRC32 checksum.
    // source: http://home.thep.lu.se/~bjorn/crc/

    static uint32_t crc32_for_byte(uint32_t r) {
        for (int j = 0; j < 8; ++j)
            r = (r & 1 ? 0 : (uint32_t)0xEDB88320L) ^ r >> 1;
        return r ^ (uint32_t)0xFF000000L;
    }

    static void crc32(const void *data, size_t n_bytes, uint32_t* crc) {
        static uint32_t table[0x100];
        if (!*table)
            for (uint32_t i = 0; i < 0x100; ++i)
                table[i] = crc32_for_byte(i);
        for (size_t i = 0; i < n_bytes; ++i)
            *crc = table[(uint8_t)*crc ^ ((uint8_t*)data)[i]] ^ *crc >> 8;
    }

    void SetupNnet();
    intel_nnet_layer_t nnet_layer ={};
    intel_nnet_type_t nnet = { 1,4, &nnet_layer };
    intel_affine_layer_t affine_layer;
    gna_device_id deviceIndex = 0;
    uint32_t rw_buffer_size = 0;
    void *memory = nullptr;
};

TEST_F(TestSimpleModel, exportSueLegacyTest)
{
    SetupNnet();

    gna_model_id model_id;
    auto status = GnaModelCreate(deviceIndex, &nnet, &model_id);
    EXPECT_EQ(status, GNA_SUCCESS);

    intel_gna_model_header model_header;
    void* dumped_model = GnaModelDump(model_id, GNA_1_0_EMBEDDED, &model_header, &status, Allocator);
    EXPECT_EQ(status, GNA_SUCCESS);
    EXPECT_NE(dumped_model, nullptr);

    model_header.rw_region_size = rw_buffer_size;

    EXPECT_EQ(sizeof(intel_gna_model_header), expectedHeaderSize);
    EXPECT_EQ(model_header.model_size, expectedModelSize);

    uint32_t headerHash = 0;
    crc32(&model_header, expectedHeaderSize, &headerHash);

    uint32_t modelHash = 0;
    crc32(dumped_model, expectedModelSize, &modelHash);

    uint32_t fileHash = headerHash;
    crc32(dumped_model, expectedModelSize, &fileHash);

    EXPECT_EQ(expected_fileHash, fileHash);
    EXPECT_EQ(expected_headerHash, headerHash);
    EXPECT_EQ(expected_modelHash, modelHash);

    Free(dumped_model);
    status = GnaFree(memory);
    status = GnaDeviceClose(deviceIndex);
}

TEST_F(TestSimpleModel, exportSueLegacyTestUsingApi2)
{
    SetupNnet();

    gna_model_id model_id;
    auto status = GnaModelCreate(deviceIndex, &nnet, &model_id);
    EXPECT_EQ(status, GNA_SUCCESS);

    Gna2ModelSueCreekHeader modelHeader;

    void * bufferLdHeader;
    void * bufferDump;

    uint32_t bufferLdHeaderSize;
    uint32_t bufferDumpSize;

    uint32_t exportConfig;

    auto status2 = Gna2ModelExportConfigCreate(Allocator, &exportConfig);
    EXPECT_EQ(status2, Gna2StatusSuccess);
    status2 = Gna2ModelExportConfigSetSource(exportConfig, deviceIndex, model_id);
    EXPECT_EQ(status2, Gna2StatusSuccess);
    status2 = Gna2ModelExportConfigSetTarget(exportConfig, Gna2DeviceVersionSueCreek);
    EXPECT_EQ(status2, Gna2StatusSuccess);

    status2 = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLegacySueCreekHeader,
        &bufferLdHeader, &bufferLdHeaderSize);

    EXPECT_EQ(status2, Gna2StatusSuccess);
    EXPECT_EQ(bufferLdHeaderSize, expectedHeaderSize);

    modelHeader = *(reinterpret_cast<Gna2ModelSueCreekHeader*>(bufferLdHeader));
    EXPECT_EQ(modelHeader.ModelSize, expectedModelSize);

    status2 = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLegacySueCreekDump,
        &bufferDump,
        &bufferDumpSize);
    EXPECT_EQ(status2, Gna2StatusSuccess);
    EXPECT_NE(bufferDump, nullptr);
    EXPECT_EQ(bufferDumpSize, expectedModelSize);

    status2 = Gna2ModelExportConfigRelease(exportConfig);
    EXPECT_EQ(status2, Gna2StatusSuccess);

    modelHeader.RwRegionSize = rw_buffer_size;

    uint32_t headerHash = 0;
    crc32(&modelHeader, expectedHeaderSize, &headerHash);

    uint32_t modelHash = 0;
    crc32(bufferDump, expectedModelSize, &modelHash);

    uint32_t fileHash = headerHash;
    crc32(bufferDump, expectedModelSize, &fileHash);

    EXPECT_EQ(expected_fileHash, fileHash);
    EXPECT_EQ(expected_headerHash, headerHash);
    EXPECT_EQ(expected_modelHash, modelHash);

    Free(bufferLdHeader);
    Free(bufferDump);
    status = GnaFree(memory);
    status = GnaDeviceClose(deviceIndex);
}

void TestSimpleModel::SetupNnet()
{
    gna_device_id deviceNumber;
    auto status = GnaDeviceGetCount(&deviceNumber);
    EXPECT_EQ(GNA_SUCCESS, status);

    status = GnaDeviceOpen(deviceIndex);
    EXPECT_EQ(GNA_SUCCESS, status);

    rw_buffer_size = ALIGN(buf_size_inputs + buf_size_outputs + buf_size_tmp_outputs, 0x1000);
    uint32_t bytes_requested = rw_buffer_size + buf_size_weights + buf_size_biases;

    uint32_t bytes_granted;

    status = GnaAlloc(bytes_requested, &bytes_granted, &memory);
    EXPECT_EQ(GNA_SUCCESS, status);
    EXPECT_NE(memory, nullptr);

    uint8_t *model_memory = (uint8_t*)memory;

    uint8_t *rw_buffers = model_memory;

    int16_t *pinned_inputs = (int16_t*)rw_buffers;
    memcpy_s(pinned_inputs, buf_size_inputs, inputs, sizeof(inputs));
    rw_buffers += buf_size_inputs;

    int16_t *pinned_outputs = (int16_t*)rw_buffers;
    rw_buffers += buf_size_outputs;

    int32_t *tmp_outputs_buffer = (int32_t*)rw_buffers;

    model_memory += rw_buffer_size;
    int16_t *weights_buffer = (int16_t*)model_memory;
    memcpy_s(weights_buffer, buf_size_weights, weights, sizeof(weights));
    model_memory += buf_size_weights;

    int32_t *biases_buffer = (int32_t*)model_memory;
    memcpy_s(biases_buffer, buf_size_biases, biases, sizeof(biases));

    intel_affine_func_t affine_func;
    affine_func.nBytesPerWeight = GNA_INT16;
    affine_func.nBytesPerBias = GNA_INT32;
    affine_func.pWeights = weights_buffer;
    affine_func.pBiases = biases_buffer;

    intel_pwl_func_t pwl;
    pwl.nSegments = 0;
    pwl.pSegments = NULL;

    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    nnet_layer.nInputColumns = nnet.nGroup;
    nnet_layer.nInputRows = 16;
    nnet_layer.nOutputColumns = nnet.nGroup;
    nnet_layer.nOutputRows = 8;
    nnet_layer.nBytesPerInput = GNA_INT16;
    nnet_layer.nBytesPerOutput = GNA_INT32;
    nnet_layer.nBytesPerIntermediateOutput = GNA_INT32;
    nnet_layer.mode = INTEL_INPUT_OUTPUT;
    nnet_layer.operation = INTEL_AFFINE;
    nnet_layer.pLayerStruct = &affine_layer;

    nnet_layer.pInputs = pinned_inputs;
    nnet_layer.pOutputsIntermediate = tmp_outputs_buffer;
    nnet_layer.pOutputs = pinned_outputs;
}
