/*
 INTEL CONFIDENTIAL
 Copyright 2019-2020 Intel Corporation.

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

#include "test-gna-api.h"

#include "gna2-model-api.h"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

static void GNA_OK_helper(const char* what, Gna2Status status)
{
    ASSERT_EQ(status, Gna2StatusSuccess) << what;
}

#define GNA_OK(what)                                    \
    {                                                   \
        const auto status = (what);                     \
        GNA_OK_helper(#what, status);                   \
    }

class TestSimpleModel : public TestGnaApi
{
protected:

    void TearDown() override
    {
        FreeAndClose();
        TestGnaApi::TearDown();
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

    const uint8_t refAdlNoMmuLd[128] = {
        0x00, 0x0A, 0x10, 0x00, 0x08, 0x00, 0x04, 0x01,     // 0x00
        0x10, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // 0x10
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,     // 0x20
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x81, 0x10, 0x00, 0x00, 0x81, 0x11, 0x00, 0x00,     // 0x30
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // 0x40
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // 0x50
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // 0x60
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     // 0x70
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, };

    bool enablePwl = false;
    const Gna2PwlSegment identityPwl[2] = {
        {(std::numeric_limits<int16_t>::min)(), (std::numeric_limits<int16_t>::min)(), 0x100},
        {0, 0, 0x100} };

    void ExpectEqualToRefAdlNoMmuLd(const uint8_t* dump, uint32_t dumpSize) const;

    const int buf_size_weights = Gna2RoundUpTo64(sizeof(weights));
    const int buf_size_inputs = Gna2RoundUpTo64(sizeof(inputs));
    const int buf_size_biases = Gna2RoundUpTo64(sizeof(biases));
    const int buf_size_outputs = Gna2RoundUpTo64(8 * 4 * 4);
    const int buf_size_identity_pwl = Gna2RoundUpTo64(sizeof(identityPwl));

    void* AllocatorAligned64(uint32_t size) {
        return _mm_malloc(size, 64);
    }

    static void * AllocatorAlignedPage(uint32_t size)
    {
        return _mm_malloc(size, 4096);
    }

    static void Free(void * mem)
    {
        return _mm_free(mem);
    }

    const uint32_t expectedModelSize = 8512;
    const uint32_t expectedHeaderSize = 64;

    const uint32_t expected_headerHash = 0xecf42aea;
    const uint32_t expected_modelHash = 0x93759a0d;
    const uint32_t expected_fileHash = 0xc8f5660a;

    // Based on
    // Simple public domain implementation of the standard CRC32 checksum.
    // source: http://home.thep.lu.se/~bjorn/crc/

    static uint32_t crc32_for_byte(uint32_t r) {
        for (int j = 0; j < 8; ++j)
        {
            r = ((r & 1) > 0 ? 0 : (uint32_t)0xEDB88320L) ^ r >> 1;
        }
        return r ^ (uint32_t)0xFF000000L;
    }

    static void crc32(const void *data, size_t n_bytes, uint32_t* crc) {
        static uint32_t table[0x100];
        if (*table == 0)
        {
            for (uint32_t i = 0; i < 0x100; ++i)
            {
                table[i] = crc32_for_byte(i);
            }
        }
        for (size_t i = 0; i < n_bytes; ++i)
        {
            *crc = table[(uint8_t)*crc ^ ((uint8_t*)data)[i]] ^ *crc >> 8;
        }
    }

    void SetupGnaMemPointers(bool setupPwl, bool setupInputsOutputs);
    void CopyDataToGnaMem(bool copyPwl, bool copyInputs) const;
    void FreeAndClose();

    Gna2Tensor inputTensor{ {2, {16,4}}, Gna2TensorModeDefault, {}, Gna2DataTypeInt16, nullptr };
    Gna2Tensor outputTensor{ {2, {8,4}}, Gna2TensorModeDefault, {}, Gna2DataTypeInt32, nullptr };
    Gna2Tensor weightTensor{ {2, {8,16}}, Gna2TensorModeDefault, {}, Gna2DataTypeInt16, nullptr };
    Gna2Tensor biasTensor{ {1, {8}}, Gna2TensorModeDefault, {}, Gna2DataTypeInt32, nullptr };
    Gna2Tensor optionalPwlTensor{ {1, {2}}, Gna2TensorModeDefault, {}, Gna2DataTypePwlSegment, nullptr };
    static const uint32_t numberOfTensors = 4;
    const Gna2Tensor* gnaOperands[numberOfTensors + 1] = { &inputTensor, &outputTensor , &weightTensor , &biasTensor, &optionalPwlTensor};

    Gna2Operation gnaOperations = { Gna2OperationTypeFullyConnectedAffine,
        gnaOperands, numberOfTensors,
        nullptr,0 };

    Gna2Model gnaModel{ 1, &gnaOperations };
    uint32_t gnaModelId = 0;
    bool minimizeRw = false;
    bool separateInputAndOutput = false;
    void SetupGnaModel();
    void SetupGnaModelSue(bool separateRO);
    void CreateGnaModel();

    void ExportSueLegacyUsingGnaApi2(Gna2ModelSueCreekHeader& modelHeader, std::vector<char>& dump);

    void PrepareExportConfig(Gna2DeviceVersion deviceVersion);
    void ExportComponent(std::vector<char> & destination, Gna2DeviceVersion deviceVersion, Gna2ModelExportComponent component);

    template<class T>
    void ExportComponentAs(T & destination, Gna2DeviceVersion deviceVersion, Gna2ModelExportComponent component)
    {
        std::vector<char> out;
        ExportComponent(out, deviceVersion, component);
        EXPECT_EQ(out.size(), sizeof(T));
        destination = *reinterpret_cast<T*>(out.data());
    }

    uint32_t exportConfig = 0;

    uint32_t deviceIndex = 0;
    uint32_t rw_buffer_size = 0;
    uint32_t ro_buffer_size = 0;
    void *memory = nullptr;
    uint32_t memorySize = 0;
    uint32_t gnamem_pinned_inputs_size = 0;
    uint32_t gnamem_pinned_outputs_size = 0;

    int16_t* gnamem_pinned_inputs = nullptr;
    int16_t* gnamem_pinned_outputs = nullptr;
    int16_t* gnamem_weights_buffer = nullptr;
    int32_t* gnamem_biases_buffer = nullptr;
    Gna2PwlSegment* gnamem_pwl_buffer = nullptr;
};
