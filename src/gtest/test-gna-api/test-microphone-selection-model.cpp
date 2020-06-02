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

    static void GnaAllocAndTag(void* & gnaMemoryOut, const uint32_t memorySize, const uint32_t memoryTag)
    {
        uint32_t granted = 0;
        ASSERT_EQ(Gna2StatusSuccess, Gna2MemoryAlloc(memorySize, &granted, &gnaMemoryOut));
        ASSERT_EQ(granted, memorySize);
        ASSERT_NE(gnaMemoryOut, nullptr);
        ASSERT_EQ(Gna2StatusSuccess, Gna2MemorySetTag(gnaMemoryOut, memoryTag));
    }

    void exportForAnna(bool outAsExternal = false, bool inputAlsoFromExternal = false, uint32_t inputExternalOffset = 0);

};


void TestTlvReader(const char* tlvBlob, uint32_t tlvSize, bool outAsExternal, std::string userSignature)
{
    std::vector<Gna2TlvType> TlvTypes =
    {
        TlvTypeLayerDescriptorArraySize,
        TlvTypeLayerDescriptorAndRoArrayData,
        TlvTypeStateData,
        TlvTypeScratchSize,
        TlvTypeExternalInputBufferSize,
        TlvTypeExternalOutputBufferSize,
        TlvTypeUserSignatureData,
   };

    for(auto type: TlvTypes)
    {
        void * value = nullptr;
        uint32_t valueSize = 0;;
        const auto tlvStatus = Gna2TlvFindInArray(tlvBlob, tlvSize, type, &valueSize, &value);
        EXPECT_EQ(tlvStatus, Gna2TlvStatusSuccess);
        if(type == TlvTypeLayerDescriptorAndRoArrayData ||
            type == TlvTypeStateData)
        {
            EXPECT_TRUE((((char*)value) - tlvBlob) % TLV_ANNA_REQUIRED_ALIGNEMENT == 0);
        }
        if(type == TlvTypeUserSignatureData)
        {
            EXPECT_TRUE(userSignature == ((char*)value));
        }
        if (!outAsExternal)
        {
            EXPECT_NE(valueSize, 0);
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

    ASSERT_EQ(Gna2StatusSuccess, Gna2OperationInitFullyConnectedBiasGrouping(&operation, PageAllocator,
        &inputOperand, &outputOperand, &weightOperand, &biasOperand, &pwlOperand, nullptr,
        &biasMode, &selectedMicrophone));
    uint32_t modelId = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelCreate(DeviceIndex, &model, &modelId));
    uint32_t exportConfigId = 0;
    ASSERT_EQ(Gna2StatusSuccess, Gna2ModelExportConfigCreate(PageAllocator, &exportConfigId));
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
        userSignature, sizeof(userSignature)
    );
    EXPECT_EQ(Gna2TlvStatusSuccess, tlvStatus);

    TestTlvReader(outTlvVector, outTlvSize, outAsExternal, userSignature);

    Free(outTlvVector);
    PageFree(exportBufferExOut);
    PageFree(exportBufferExIn);
    PageFree(exportBufferScratch);
    PageFree(exportBufferState);
    PageFree(exportBufferRO);
    PageFree(exportBufferLda);
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
