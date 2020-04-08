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

#include "test-model-export-legacy-sue.h"

#include "test-activation-helper.h"
#include "test-gna-api.h"

#include "gna-api.h"
#include "gna-api-dumper.h"

#include "gna2-device-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-api.h"
#include "gna2-model-export-impl.h"

#include <cstring>
#include <gtest/gtest.h>
#include <fstream>

#include "HardwareModelNoMMU.h"

#if !defined(__STDC_LIB_EXT1__) && !defined(memcpy_s)
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) memcpy(_Destination, _Source, _SourceSize)
#endif

void TestSimpleModel::FreeAndClose2()
{
    EXPECT_EQ(Gna2MemoryFree(memory), Gna2StatusSuccess);
    EXPECT_EQ(Gna2DeviceClose(deviceIndex), Gna2StatusSuccess);
}

void TestSimpleModel::PrepareExportConfig(Gna2DeviceVersion deviceVersion)
{
    GNA_OK(Gna2ModelExportConfigCreate(AllocatorAlignedPage, &exportConfig));
    GNA_OK(Gna2ModelExportConfigSetSource(exportConfig, 0, gnaModelId))
    GNA_OK(Gna2ModelExportConfigSetTarget(exportConfig, deviceVersion));
}

void TestSimpleModel::ExportComponent(std::vector<char> & destination, Gna2DeviceVersion deviceVersion, Gna2ModelExportComponent component)
{
    PrepareExportConfig(deviceVersion);

    void * ldNoMmu = nullptr;
    uint32_t ldNoMmuSize = 0;

    GNA_OK(Gna2ModelExport(exportConfig,
        component,
        &ldNoMmu, &ldNoMmuSize));
    EXPECT_NE(ldNoMmu, nullptr);
    destination.resize(ldNoMmuSize);
    memcpy(destination.data(), static_cast<char*>(ldNoMmu), ldNoMmuSize);
    Free(ldNoMmu);
    GNA_OK(Gna2ModelExportConfigRelease(exportConfig));
}

void TestSimpleModel::ExportSueLegacyUsingGnaApi2(Gna2ModelSueCreekHeader& modelHeader, std::vector<char>& dump)
{
    ExportComponentAs(modelHeader, Gna2DeviceVersionEmbedded1_0, Gna2ModelExportComponentLegacySueCreekHeader);

    ExportComponent(dump, Gna2DeviceVersionEmbedded1_0, Gna2ModelExportComponentLegacySueCreekDump);
    EXPECT_EQ(dump.size(), expectedModelSize);
}

void TestSimpleModel::ExpectEqualToRefAdlNoMmuLd(const uint8_t* dump, uint32_t dumpSize) const
{
    ExpectMemEqual(dump, dumpSize, refAdlNoMmuLd, sizeof(refAdlNoMmuLd));
}

void TestSimpleModel::SetupGnaMemPointers(const bool setupPwl, const bool setupInputsOutputs)
{
    if (setupInputsOutputs)
    {
        uint8_t *rw_buffers = (uint8_t*)memory;
        gnamem_pinned_inputs = (int16_t*)rw_buffers;
        rw_buffers += buf_size_inputs;
        gnamem_pinned_outputs = (int16_t*)rw_buffers;
        rw_buffers += buf_size_outputs;
        gnamem_tmp_outputs_buffer = (int32_t*)rw_buffers;
    }

    uint8_t *model_memory = (uint8_t*)memory;
    model_memory += rw_buffer_size;
    gnamem_weights_buffer = (int16_t*)model_memory;
    model_memory += buf_size_weights;
    gnamem_biases_buffer = (int32_t*)model_memory;
    if (setupPwl)
    {
        model_memory += buf_size_biases;
        gnamem_pwl_buffer = (Gna2PwlSegment*)model_memory;
    }
}

void TestSimpleModel::CopyDataToGnaMem(const bool copyPwl, const bool copyInputs)
{
    if (copyInputs)
    {
        memcpy_s(gnamem_pinned_inputs, buf_size_inputs, inputs, sizeof(inputs));
    }
    memcpy_s(gnamem_weights_buffer, buf_size_weights, weights, sizeof(weights));
    memcpy_s(gnamem_biases_buffer, buf_size_biases, biases, sizeof(biases));

    if (copyPwl)
    {
        memcpy_s(gnamem_pwl_buffer, buf_size_identity_pwl, identityPwl, sizeof(identityPwl));
    }
}

void TestSimpleModel::SetupNnet()
{
   /* uint32_t deviceNumber;
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

    SetupGnaMemPointers(false, true);

    CopyDataToGnaMem(false, true);

    intel_affine_func_t affine_func;
    affine_func.nBytesPerWeight = GNA_INT16;
    affine_func.nBytesPerBias = GNA_INT32;
    affine_func.pWeights = gnamem_weights_buffer;
    affine_func.pBiases = gnamem_biases_buffer;

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

    nnet_layer.pInputs = gnamem_pinned_inputs;
    nnet_layer.pOutputsIntermediate = gnamem_tmp_outputs_buffer;
    nnet_layer.pOutputs = gnamem_pinned_outputs;*/
}

void TestSimpleModel::SetupGnaModel()
{
    uint32_t deviceCount;
    auto status = Gna2DeviceGetCount(&deviceCount);
    EXPECT_EQ(Gna2StatusSuccess, status);
    EXPECT_GT(deviceCount, deviceIndex);

    status = Gna2DeviceOpen(deviceIndex);

    EXPECT_EQ(Gna2StatusSuccess, status);

    rw_buffer_size = ALIGN(buf_size_inputs + buf_size_outputs + buf_size_tmp_outputs, 0x1000);
    if(minimizeRw)
    {
        rw_buffer_size = buf_size_inputs + buf_size_outputs;
    }
    if(separateInputAndOutput)
    {
        rw_buffer_size = 0;
        status = Gna2MemoryAlloc(buf_size_inputs, &gnamem_pinned_inputs_size, reinterpret_cast<void**>(&gnamem_pinned_inputs));
        EXPECT_EQ(Gna2StatusSuccess, status);
        status = Gna2MemoryAlloc(buf_size_outputs, &gnamem_pinned_outputs_size, reinterpret_cast<void**>(&gnamem_pinned_outputs));
        EXPECT_EQ(Gna2StatusSuccess, status);
        status = Gna2MemorySetTag(gnamem_pinned_inputs, GNA::HardwareModelNoMMU::MemoryTagInput);
        EXPECT_EQ(Gna2StatusSuccess, status);
        status = Gna2MemorySetTag(gnamem_pinned_outputs, GNA::HardwareModelNoMMU::MemoryTagOutput);
        EXPECT_EQ(Gna2StatusSuccess, status);
    }
    uint32_t bytes_requested = rw_buffer_size + buf_size_weights + buf_size_biases;
    if(enablePwl)
    {
        bytes_requested += buf_size_identity_pwl;
    }
    status = Gna2MemoryAlloc(bytes_requested, &memorySize, &memory);
    EXPECT_EQ(Gna2StatusSuccess, status);
    EXPECT_NE(memory, nullptr);
    const bool copyInputs = !minimizeRw;
    const bool setupInOut = !separateInputAndOutput;

    SetupGnaMemPointers(enablePwl, setupInOut);
    CopyDataToGnaMem(enablePwl, copyInputs);

    weightTensor.Data = gnamem_weights_buffer;
    biasTensor.Data = gnamem_biases_buffer;
    inputTensor.Data = gnamem_pinned_inputs;
    outputTensor.Data = gnamem_pinned_outputs;
    if(enablePwl)
    {
        optionalPwlTensor.Data = gnamem_pwl_buffer;
        gnaOperations.NumberOfOperands = 5;
    }
    CreateGnaModel();
}

void TestSimpleModel::CreateGnaModel()
{
    const auto status = Gna2ModelCreate(deviceIndex, &gnaModel, &gnaModelId);
    EXPECT_EQ(status, Gna2StatusSuccess);
}

TEST_F(TestSimpleModel, exportSueLegacyTestUsingApi2)
{
    /*SetupNnet();

    auto status = GnaModelCreate(deviceIndex, &nnet, &gnaModelId);
    EXPECT_EQ(status, GNA_SUCCESS);

    Gna2ModelSueCreekHeader modelHeader;
    std::vector<char> dump;
    ExportSueLegacyUsingGnaApi2(modelHeader, dump);
    EXPECT_EQ(modelHeader.ModelSize, expectedModelSize);

    modelHeader.RwRegionSize = rw_buffer_size;

    uint32_t headerHash = 0;
    crc32(&modelHeader, expectedHeaderSize, &headerHash);

    uint32_t modelHash = 0;
    crc32(dump.data(), expectedModelSize, &modelHash);

    uint32_t fileHash = headerHash;
    crc32(dump.data(), expectedModelSize, &fileHash);

    EXPECT_EQ(expected_fileHash, fileHash);
    EXPECT_EQ(expected_headerHash, headerHash);
    EXPECT_EQ(expected_modelHash, modelHash);

    FreeAndClose2();*/
}

TEST_F(TestSimpleModel, exportNoMmuApi2)
{
    SetupGnaModel();

    PrepareExportConfig(Gna2DeviceVersionEmbedded3_0);

    void * bufferLd;
    uint32_t bufferLdSize;
    auto status2 = Gna2ModelExport(exportConfig,
        Gna2ModelExportComponentLayerDescriptors,
        &bufferLd, &bufferLdSize);
    EXPECT_EQ(status2, Gna2StatusSuccess);

    ExpectEqualToRefAdlNoMmuLd(static_cast<uint8_t*>(bufferLd), bufferLdSize);

    status2 = Gna2ModelExportConfigRelease(exportConfig);
    EXPECT_EQ(status2, Gna2StatusSuccess);

    FreeAndClose2();
}
