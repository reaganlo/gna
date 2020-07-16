/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "test-gna-api.h"

#include "test-activation-helper.h"

#include "HardwareModelNoMMU.h"

#include "gna2-capability-api.h"
#include "gna2-model-export-api.h"

#include "gna2-tlv-anna-writer.h"
#include "gna2-tlv-anna-reader.h"

#include <cstdint>


class TestMicrophoneSelectionModel : public TestGnaModel
{
public:
    const uint32_t numberOfMics = 4;
    const uint32_t numberOfElementsFromMic = 1024;
    const uint32_t externalBufferSize = sizeof(int32_t) * numberOfMics * numberOfElementsFromMic;
    void* mockedExternalBuffer = nullptr;
    const uint32_t minimalMBInputVectorSize = 8;
    const uint32_t inputGroups = 1;
    // For weights and inputs, size determined on the weights basis as bigger
    const uint32_t memoryZeroedSize = numberOfElementsFromMic * minimalMBInputVectorSize * sizeof(int16_t);

    Gna2Operation operation = {};

    Gna2BiasMode biasMode = Gna2BiasModeGrouping;
    uint32_t selectedMicrophone = 1;

    Gna2Model model{ 1, &operation };

    uint8_t expectedLda[128] = {
        0x09, 0x2a, 0x08, 0x00, 0x00, 0x04, 0x01, 0x01, 0x08, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x81, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x81, 0x40, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    const uint32_t expectedLdaSize = sizeof(expectedLda);

    void GnaAllocAndTag(void* & gnaMemoryOut, const uint32_t memorySize, const uint32_t memoryTag)
    {
        uint32_t granted = 0;
        ASSERT_EQ(Gna2StatusSuccess, Gna2MemoryAlloc(memorySize, &granted, &gnaMemoryOut));
        ASSERT_EQ(granted, memorySize);
        ASSERT_NE(gnaMemoryOut, nullptr);
        ASSERT_EQ(Gna2StatusSuccess, Gna2MemorySetTag(gnaMemoryOut, memoryTag));
        gnaMems.push_back(gnaMemoryOut);
    }

    void GnaFreeAll()
    {
        for(auto m: gnaMems)
        {
            ASSERT_EQ(Gna2StatusSuccess, Gna2MemoryFree(m));
        }
        gnaMems.clear();
    }
    void exportForAnna(bool outAsExternal = false, bool inputAlsoFromExternal = false, uint32_t inputExternalOffset = 0);
private:
    std::vector<void* > gnaMems;
};


void TestTlvReader(const char* tlvBlob, uint32_t tlvSize, bool outAsExternal, std::string userSignature)
{
    EXPECT_EQ(Gna2TlvStatusSuccess, Gna2TlvVerifyVersionAndCohesion(tlvBlob, tlvSize));
    EXPECT_EQ(Gna2TlvStatusVersionNotFound, Gna2TlvVerifyVersionAndCohesion(tlvBlob+12, tlvSize-12));
    EXPECT_EQ(Gna2TlvStatusTlvReadError, Gna2TlvVerifyVersionAndCohesion(tlvBlob, tlvSize-1));
    EXPECT_EQ(Gna2TlvStatusTlvReadError, Gna2TlvVerifyVersionAndCohesion(tlvBlob, tlvSize+1));
    EXPECT_EQ(Gna2TlvStatusTlvReadError, Gna2TlvVerifyVersionAndCohesion(tlvBlob+1, tlvSize-1));

    void * value = nullptr;
    uint32_t valueLength = 0;
    EXPECT_EQ(Gna2TlvStatusNotFound, Gna2TlvFindInArray(tlvBlob, tlvSize, 0x01010101, &valueLength, &value));

    std::vector<Gna2TlvType> TlvTypes =
    {
        Gna2TlvTypeLayerDescriptorArraySize,
        Gna2TlvTypeLayerDescriptorAndRoArrayData,
        Gna2TlvTypeStateData,
        Gna2TlvTypeScratchSize,
        Gna2TlvTypeExternalInputBufferSize,
        Gna2TlvTypeExternalOutputBufferSize,
        Gna2TlvTypeUserData,
        Gna2TlvTypeTlvVersion,
        Gna2TlvTypeGnaLibraryVersionString,
    };

    for(auto type: TlvTypes)
    {
        value = nullptr;
        valueLength = 0;
        const auto tlvStatus = Gna2TlvFindInArray(tlvBlob, tlvSize, type, &valueLength, &value);
        EXPECT_EQ(tlvStatus, Gna2TlvStatusSuccess);
        if(type == Gna2TlvTypeLayerDescriptorAndRoArrayData ||
            type == Gna2TlvTypeStateData)
        {
            EXPECT_TRUE((((char*)value) - tlvBlob) % GNA2_TLV_ANNA_REQUIRED_ALIGNEMENT == 0);
        }
        if(type == Gna2TlvTypeUserData)
        {
            EXPECT_TRUE(userSignature == ((char*)value));
        }
        if (!outAsExternal || type != Gna2TlvTypeStateData)
        {
            EXPECT_NE(valueLength, 0);
        }
        if (outAsExternal && type == Gna2TlvTypeStateData)
        {
            EXPECT_EQ(valueLength, 0);
        }
        if(type == Gna2TlvTypeGnaLibraryVersionString)
        {
            char buffer[1024] = {};
            EXPECT_EQ(Gna2TlvStatusSuccess, Gna2GetLibraryVersion(buffer, sizeof(buffer)));
            EXPECT_EQ(std::string(buffer).size() + 1, (size_t)valueLength);
            EXPECT_EQ(std::string(buffer), std::string(((char*)value)));
        }
        if(type == Gna2TlvTypeTlvVersion)
        {
            EXPECT_EQ(GNA2_TLV_VERSION, 1);
            EXPECT_EQ(*(uint32_t*)value, GNA2_TLV_VERSION);
        }
        EXPECT_NE(value, (void*)NULL);
    }

}

void TestMicrophoneSelectionModel::exportForAnna(bool outAsExternal, bool inputAlsoFromExternal, uint32_t inputExternalOffset)
{
    const auto& conversionPwl = TestActivationHelper::GetConversionPwl();
    const uint32_t memoryPwlSize = conversionPwl.size() * sizeof(Gna2PwlSegment);

    void * gnaMemoryRO;
    const uint32_t memoryROSize = ALIGN64(memoryZeroedSize) + ALIGN64(memoryPwlSize);

    void * gnaMemoryOutput;
    const uint32_t memoryOutputSize = numberOfElementsFromMic * inputGroups * sizeof(int16_t);

    GnaAllocAndTag(gnaMemoryRO, memoryROSize, GNA::HardwareModelNoMMU::MemoryTagReadOnly);

    if (outAsExternal)
    {
        GnaAllocAndTag(gnaMemoryOutput, memoryOutputSize, GNA::HardwareModelNoMMU::MemoryTagExternalBufferOutput);
    }
    else
    {
        GnaAllocAndTag(gnaMemoryOutput, memoryOutputSize, GNA::HardwareModelNoMMU::MemoryTagState);
    }

    GnaAllocAndTag(mockedExternalBuffer, externalBufferSize, GNA::HardwareModelNoMMU::MemoryTagExternalBufferInput);



    void * gnaMemoryZeroed = gnaMemoryRO;
    void * gnaMemoryPwl = static_cast<uint8_t*>(gnaMemoryRO) + ALIGN64(memoryZeroedSize);
    std::copy_n(conversionPwl.begin(), conversionPwl.size(), static_cast<Gna2PwlSegment*>(gnaMemoryPwl));
    std::fill_n(static_cast<uint8_t*>(gnaMemoryZeroed), memoryZeroedSize, 0);

    auto inputOperand = Gna2TensorInit2D(minimalMBInputVectorSize, inputGroups, Gna2DataTypeInt16, gnaMemoryZeroed);

    if (inputAlsoFromExternal)
    {
        inputOperand.Data = static_cast<uint8_t*>(mockedExternalBuffer) + inputExternalOffset;
        inputOperand.Mode = Gna2TensorModeExternalBuffer;
    }

    auto outputOperand = Gna2TensorInit2D(numberOfElementsFromMic, inputGroups, Gna2DataTypeInt16, gnaMemoryOutput);
    if (outAsExternal)
    {
        outputOperand.Mode = Gna2TensorModeExternalBuffer;
    }
    auto weightOperand = Gna2TensorInit2D(numberOfElementsFromMic, minimalMBInputVectorSize, Gna2DataTypeInt16, gnaMemoryZeroed);
    auto biasOperand = Gna2TensorInit2D(numberOfElementsFromMic, numberOfMics, Gna2DataTypeInt32, mockedExternalBuffer);
    biasOperand.Mode = Gna2TensorModeExternalBuffer;
    auto pwlOperand = Gna2TensorInit1D(conversionPwl.size(), Gna2DataTypePwlSegment, gnaMemoryPwl);

    ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitFullyConnectedBiasGrouping(&operation, AlignedAllocator,
        &inputOperand, &outputOperand, &weightOperand, &biasOperand, &pwlOperand, nullptr,
        &biasMode, &selectedMicrophone));
    uint32_t modelId = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelCreate(DeviceIndex, &model, &modelId));
    uint32_t exportConfigId = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigCreate(AlignedAllocator, &exportConfigId));
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigSetSource(exportConfigId, DeviceIndex, modelId));
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigSetTarget(exportConfigId, Gna2DeviceVersionEmbedded3_1));
    void * exportBufferLda;
    uint32_t exportBufferSizeLda = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentLayerDescriptors,
        &exportBufferLda, &exportBufferSizeLda));

    ExpectMemEqual(static_cast<uint8_t*>(exportBufferLda), exportBufferSizeLda, expectedLda, expectedLdaSize);

    void * exportBufferRO;
    uint32_t exportBufferSizeRO = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentReadOnlyDump,
        &exportBufferRO, &exportBufferSizeRO));

    void * exportBufferState;
    uint32_t exportBufferSizeState = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentStateDump,
        &exportBufferState, &exportBufferSizeState));

    void * exportBufferScratch;
    uint32_t exportBufferSizeScratch = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentScratchDump,
        &exportBufferScratch, &exportBufferSizeScratch));

    void * exportBufferExIn;
    uint32_t exportBufferExInSize = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentExternalBufferInputDump,
        &exportBufferExIn, &exportBufferExInSize));

    void * exportBufferExOut;
    uint32_t exportBufferExOutSize = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentExternalBufferOutputDump,
        &exportBufferExOut, &exportBufferExOutSize));

    void * exportBufferIn;
    uint32_t exportBufferExSize = 0;
    ASSERT_EQ(Gna2StatusMemoryBufferInvalid, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentInputDump,
        &exportBufferIn, &exportBufferExSize));

    void * exportBufferOut;
    uint32_t exportBufferOutSize = 0;
    ASSERT_EQ(Gna2StatusMemoryBufferInvalid, Gna2ModelExport(exportConfigId,
        Gna2ModelExportComponentOutputDump,
        &exportBufferOut, &exportBufferOutSize));

    char gnaLibraryVersionCString[32] = "UNKNOWN";
    ASSERT_EQ(Gna2StatusSuccess, Gna2GetLibraryVersion(gnaLibraryVersionCString, sizeof(gnaLibraryVersionCString)));
    char *outTlvVector = nullptr;
    uint32_t outTlvSize = 0;
    const char userSignature[] = "ANNA Microphone Selection Model";
    const auto tlvStatus = Gna2ExportAnnaTlv(
        Allocator,
        &outTlvVector,
        &outTlvSize,
        static_cast<char*>(exportBufferLda), exportBufferSizeLda,
        static_cast<char*>(exportBufferRO), exportBufferSizeRO,
        static_cast<char*>(exportBufferState), exportBufferSizeState,
        exportBufferSizeScratch,
        exportBufferExInSize,
        exportBufferExOutSize,
        gnaLibraryVersionCString,
        userSignature, sizeof(userSignature)
    );
    EXPECT_EQ(Gna2TlvStatusSuccess, tlvStatus);

    TestTlvReader(outTlvVector, outTlvSize, outAsExternal, userSignature);

    Free(outTlvVector);
    AlignedFree(exportBufferExOut);
    AlignedFree(exportBufferExIn);
    AlignedFree(exportBufferScratch);
    AlignedFree(exportBufferState);
    AlignedFree(exportBufferRO);
    AlignedFree(exportBufferLda);
}

TEST_F(TestMicrophoneSelectionModel, exportForAnnaDefault)
{
    exportForAnna();
}

TEST_F(TestMicrophoneSelectionModel, exportForAnnaOutAsExternal)
{
    expectedLda[0x1] |= 1 << 7; // set to use Output as external
    expectedLda[0x24] &= 0xFC;  // reset BAR index in Output buffer
    exportForAnna(true);
}

TEST_F(TestMicrophoneSelectionModel, exportForAnnaOutInAsExternal)
{
    const uint32_t additionalOffset = 64 * 11;  //sample additional offset for input
    expectedLda[0x1] |= 1 << 7; // set to use Output as external
    expectedLda[0x1] |= 1 << 6; // set to use Input as external
    expectedLda[0x24] &= 0xFC;  // reset BAR index in Output buffer

    // reset BAR index in Input buffer and set proper offset
    expectedLda[0x20] = static_cast<uint8_t>(additionalOffset & 0xff);
    expectedLda[0x21] = static_cast<uint8_t>((additionalOffset >> 8) & 0xff);
    exportForAnna(true, true, additionalOffset);
}
class TestCopyExportModel : public TestMicrophoneSelectionModel
{
    Gna2Operation operationArray[5] = {};

    uint8_t expectedLda[128 * 5] = {
    0x12, 0xc8, 0x00, 0x04, 0x00, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x12, 0x88, 0x00, 0x04, 0x00, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x12, 0x48, 0x00, 0x04, 0x00, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x12, 0x08, 0x00, 0x04, 0x00, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x2a, 0x08, 0x00, 0x03, 0x00, 0x06, 0x01, 0x08, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x01, 0x00, 0x00, 0x03, 0x02, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x00, 0x00, 0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x07, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

public:
    void run(
        Gna2ThresholdCondition *thresholdCondition = nullptr,
        Gna2ThresholdMode *thresholdMode = nullptr,
        Gna2ThresholdMask *thresholdMask = nullptr)
    {
        void * gnaMemoryOutput = nullptr;
        void * gnaMemoryExtra = nullptr;
        void * gnaMemoryExtraRO = nullptr;
        GnaAllocAndTag(mockedExternalBuffer, externalBufferSize, GNA::HardwareModelNoMMU::MemoryTagExternalBufferInput);
        GnaAllocAndTag(gnaMemoryOutput, 1 << 12, GNA::HardwareModelNoMMU::MemoryTagExternalBufferOutput);
        GnaAllocAndTag(gnaMemoryOutput, 1 << 12, GNA::HardwareModelNoMMU::MemoryTagExternalBufferOutput);
        GnaAllocAndTag(gnaMemoryExtra, 1 << 12, GNA::HardwareModelNoMMU::MemoryTagState);
        GnaAllocAndTag(gnaMemoryExtraRO, 1 << 12, GNA::HardwareModelNoMMU::MemoryTagReadOnly);
        auto inputOperand = Gna2TensorInit2D(inputGroups, numberOfElementsFromMic, Gna2DataTypeInt16, mockedExternalBuffer);
        auto outputOperand = Gna2TensorInit2D(inputGroups, numberOfElementsFromMic / 2, Gna2DataTypeInt16, gnaMemoryOutput);
        auto inputNormal = Gna2TensorInit2D(inputGroups, numberOfElementsFromMic, Gna2DataTypeInt16, gnaMemoryExtra);
        auto outputNormal = Gna2TensorInit2D(inputGroups, numberOfElementsFromMic/2, Gna2DataTypeInt16, gnaMemoryExtra);
        inputOperand.Mode = Gna2TensorModeExternalBuffer;
        outputOperand.Mode = Gna2TensorModeExternalBuffer;
        copyShape = Gna2ShapeInit2D(inputGroups, numberOfElementsFromMic / 4);
        ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitCopy(operationArray, AlignedAllocator,
            &inputOperand, &outputOperand, &copyShape));
        ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitCopy(operationArray+1, AlignedAllocator,
            &inputNormal, &outputOperand, &copyShape));
        ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitCopy(operationArray+2, AlignedAllocator,
            &inputOperand, &outputNormal, &copyShape));
        ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitCopy(operationArray+3, AlignedAllocator,
            &inputNormal, &outputNormal, &copyShape));
        uint8_t* ptr = (uint8_t*)gnaMemoryExtraRO + 128;
        auto inputAffineOperand = Gna2TensorInit2D(8, 6, Gna2DataTypeInt16, (uint8_t*)gnaMemoryExtra+256);
        auto outputAffineOperand = Gna2TensorInit2D(3, 6, Gna2DataTypeInt16, (uint8_t*)gnaMemoryExtra+512);
        auto weightAffineOperand = Gna2TensorInit2D(3, 8, Gna2DataTypeInt16, ptr+256*2);
        auto biasAffineOperand = Gna2TensorInit1D(3, Gna2DataTypeInt32, ptr+256*3);
        auto pwlAffineOperand = Gna2TensorInit1D(11, Gna2DataTypePwlSegment, ptr+256*4);
        ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitFullyConnectedAffine(operationArray + 4, AlignedAllocator,
            &inputAffineOperand,
            &outputAffineOperand,
            &weightAffineOperand,
            &biasAffineOperand,
            &pwlAffineOperand));
        if (thresholdCondition || thresholdMode || thresholdMask)
        {
            AlignedFree(operationArray[4].Parameters);
            operationArray[4].Parameters = static_cast<void **>(AlignedAllocator(sizeof(void*[3])));
            operationArray[4].Parameters[0] = thresholdCondition;
            operationArray[4].Parameters[1] = thresholdMode;
            operationArray[4].Parameters[2] = thresholdMask;
            operationArray[4].NumberOfParameters = 3;
            operationArray[4].NumberOfOperands = 5;
            operationArray[4].Type = Gna2OperationTypeThreshold;
        }
        uint32_t modelId = 0;
        model.Operations = operationArray;
        model.NumberOfOperations = 5;
        ASSERT_EQ(Gna2StatusSuccess, Gna2ModelCreate(DeviceIndex, &model, &modelId));
        uint32_t exportConfigId = 0;
        ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigCreate(AlignedAllocator, &exportConfigId));
        ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigSetSource(exportConfigId, DeviceIndex, modelId));
        ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigSetTarget(exportConfigId, Gna2DeviceVersionEmbedded3_1));
        void * exportBufferLda;
        uint32_t exportBufferSizeLda = 0;
        ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExport(exportConfigId,
            Gna2ModelExportComponentLayerDescriptors,
            &exportBufferLda, &exportBufferSizeLda));

        ExpectMemEqual(static_cast<uint8_t*>(exportBufferLda), exportBufferSizeLda, expectedLda, sizeof(expectedLda));
        GnaFreeAll();
        for(auto& layer: operationArray)
        {
            if(layer.Operands!=nullptr)
            {
                AlignedFree(layer.Operands);
            }
            if(layer.Parameters!=nullptr)
            {
                AlignedFree(layer.Parameters);
            }
            layer = {};
        }
    }

    void PatchRunRestore(Gna2ThresholdCondition condition, Gna2ThresholdMode operationMode, Gna2ThresholdMask interruptMask)
    {
        const uint8_t AFFINE_TH_NNOP = 0x03;
        const int NNOP_OFFSET = 128 * 4;
        const int NNFLAGSEXT_OFFSET = NNOP_OFFSET + 0xB;
        const uint8_t LDA_0xB_Mask = ((condition << 3) + (operationMode << 1) + interruptMask) << 4;
        const auto temp_nnop = expectedLda[NNOP_OFFSET];
        expectedLda[NNOP_OFFSET] = AFFINE_TH_NNOP;
        const auto temp_nnflagsext = expectedLda[NNFLAGSEXT_OFFSET];
        expectedLda[NNFLAGSEXT_OFFSET] |= LDA_0xB_Mask;
        run(&condition, &operationMode, &interruptMask);
        expectedLda[NNOP_OFFSET] = temp_nnop;
        expectedLda[NNFLAGSEXT_OFFSET] = temp_nnflagsext;
    }

    Gna2Shape copyShape = {};
};

TEST_F(TestCopyExportModel, simpleCopy)
{
    run();
}

TEST_F(TestCopyExportModel, simpleThreshold)
{
    for (auto condition : std::set<Gna2ThresholdCondition>{
        Gna2ThresholdConditionScoreNegative,
        Gna2ThresholdConditionScoreNotNegative })
    {
        for (auto mode : std::set<Gna2ThresholdMode>{
            Gna2ThresholdModeContinueNever,
            Gna2ThresholdModeContinueOnThresholdMet,
            Gna2ThresholdModeContinueOnThresholdNotMet,
            Gna2ThresholdModeContinueAlways })
        {
            for (auto mask : std::set<Gna2ThresholdMask>{
                Gna2ThresholdMaskInterruptSend,
                Gna2ThresholdMaskInterruptNotSend })
            {
                PatchRunRestore(condition, mode, mask);
            }
        }
    }
}

TEST_F(TestGnaApi, Gna2TlvVerifyVersionAndCohesion_Success)
{
    EXPECT_EQ(Gna2TlvStatusSuccess, Gna2TlvVerifyVersionAndCohesion("TLVV\4\0\0\0\1\0\0\0", 12));
}

TEST_F(TestGnaApi, Gna2TlvVerifyVersionAndCohesion_VersionNotSupported)
{
    EXPECT_EQ(Gna2TlvStatusVersionNotSupported, Gna2TlvVerifyVersionAndCohesion("TLVV\3\0\0\0\1\0\0\0", 11));
    EXPECT_EQ(Gna2TlvStatusVersionNotSupported, Gna2TlvVerifyVersionAndCohesion("TLVV\4\0\0\0\2\0\0\0", 12));
    EXPECT_EQ(Gna2TlvStatusVersionNotSupported, Gna2TlvVerifyVersionAndCohesion("TLVV\10\0\0\0\2\0\0\0\0\0\0\0", 16));
}
extern "C" int tlv_test_c_failed(const char*, unsigned);
TEST_F(TestGnaApi, Gna2Tlv_tlv_test_c_failed)
{
    EXPECT_EQ(0, tlv_test_c_failed("TLVV\4\0\0\0\1\0\0\0", 12));
}

TEST_F(TestGnaApi, Gna2Tlv_tlv_test_c_1)
{
    EXPECT_EQ(1, tlv_test_c_failed("TLVV\4\0\0\0\1\0\0\0", 11));
    EXPECT_EQ(1, tlv_test_c_failed("TLVv\4\0\0\0\1\0\0\0", 12));
    EXPECT_EQ(1, tlv_test_c_failed("TLVV\5\0\0\0\1\0\0\0", 12));
}

TEST_F(TestGnaApi, Gna2Tlv_tlv_test_c_2)
{
    EXPECT_EQ(2, tlv_test_c_failed("TLVV\10\0\0\0\1\0\0\0\0\0\0\0", 16));
}

TEST_F(TestGnaApi, Gna2Tlv_tlv_test_c_3)
{
    EXPECT_EQ(3, tlv_test_c_failed("TLVV\4\0\0\0\2\0\0\0", 12));
}
