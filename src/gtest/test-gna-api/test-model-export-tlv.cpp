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

#include "test-model-export-legacy-sue.h"

#include "gna2-api.h"
#include "gna2-capability-api.h"
#include "gna2-model-export-api.h"

#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <ostream>
#include <sstream>
#include <vector>

#define TlvLType uint32_t
static constexpr uint32_t lInTlvSize = sizeof(TlvLType);

void TlvWriteL(std::vector<char> & outVector, TlvLType l) {
    outVector.insert(outVector.end(), reinterpret_cast<char*>(&l), reinterpret_cast<char*>(&l) + sizeof(l));
}

void TlvWriteT(std::vector<char> & outVector, const std::string& type)
{
    for (unsigned i = 0; i < 4; i++)
    {
        outVector.push_back(i < type.size() ? type[i] : '\0');
    }
}

const std::vector<uint8_t> refLDFromTlvNoMmuApi2 = {
0x00, 0x3A, 0x10, 0x00, 0x08, 0x00, 0x04, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x81, 0x01, 0x00, 0x00, 0x81, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC1, 0x02, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

const std::vector<uint8_t> refLDFromTlvNoMmuApi2_TagMemory = {
0x00, 0x3A, 0x10, 0x00, 0x08, 0x00, 0x04, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x81, 0x00, 0x00, 0x00, 0x81, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC1, 0x01, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

class TestSimpleModelTlv : public TestSimpleModel
{
protected:
    struct DebugExtra
    {
        char extraString[24];
        uint32_t InputElementSize;
        uint32_t OutputElementSize;
        uint32_t NumberOfInputNodes;
        uint32_t NumberOfOutputNodes;
    };

    std::vector<char> Gna3DumpNoMmuTlv();

    static void ExpectLdEqual(const std::vector<char>& tlv, const std::vector<uint8_t>& ref)
    {
        ExpectMemEqual(reinterpret_cast<const uint8_t*>(tlv.data()) + 0x5C,
            128,
            ref.data(),
            static_cast<uint32_t>(ref.size()));
    }
};

std::vector<char> TestSimpleModelTlv::Gna3DumpNoMmuTlv()
{
    std::vector<char> outVector;
    std::stringstream debugStream;
    debugStream << "test-model-export-tlv.cpp ";
    debugStream << " GnaLibrary ";
    char version[32];
    auto const status = Gna2GetLibraryVersion(version, 32);
    if (Gna2StatusIsSuccessful(status))
    {
        debugStream << version;
    }
    else
    {
        debugStream << "UNKNOWN";
    }


    auto debugString = debugStream.str();
    debugString.resize(Gna2RoundUp(static_cast<uint32_t>(debugString.size()), 4), ' ');

    DebugExtra dbgExtra =
    {
        "Gna2ModelSueCreekHeader",
        Gna2DataTypeGetSize(inputTensor.Type),
        Gna2DataTypeGetSize(outputTensor.Type),
        inputTensor.Shape.Dimensions[0],
        outputTensor.Shape.Dimensions[0]
    };

    const uint32_t additionalScratchBufferSize = 0x2000;

    std::vector<char> ldaData;
    ExportComponent(ldaData, Gna2DeviceVersionEmbedded3_0, Gna2ModelExportComponentLayerDescriptors);

    const uint32_t tlvHeaderSize = sizeof(char[4]) + lInTlvSize;
    const uint32_t tlvTwoHeadersSize = tlvHeaderSize * 2;

    const uint32_t sizeInTlvSize = tlvHeaderSize + lInTlvSize;
    const uint32_t recordWithSizeInTlvSize = tlvHeaderSize + sizeInTlvSize;

    const std::vector<std::pair<const std::string, uint32_t> > recordsWithSizeOnly = {
        { "IN", dbgExtra.InputElementSize * dbgExtra.NumberOfInputNodes },
        { "OUT", dbgExtra.OutputElementSize * dbgExtra.NumberOfOutputNodes},
        { "SCRA", additionalScratchBufferSize} };

    // TODO: GNA2: remove DBG
    const std::vector < std::tuple<const std::string, uint32_t, const char *> > recordsWithDataOnly = {
        std::make_tuple("LDA", static_cast<uint32_t>(ldaData.size()), static_cast<const char*>(ldaData.data())),
        std::make_tuple("RORW", static_cast<uint32_t>(memorySize), static_cast<const char*>(memory)),
        std::make_tuple("DBG", static_cast<uint32_t>(sizeof(dbgExtra)), reinterpret_cast<const char*>(&dbgExtra)),
        std::make_tuple("DBG", static_cast<uint32_t>(debugString.size()), static_cast<const char*>(debugString.data())) };

    auto allGNAMComponentsInTlvSize = static_cast<uint32_t>(recordWithSizeInTlvSize * recordsWithSizeOnly.size());
    for (const auto& record : recordsWithDataOnly) {
        allGNAMComponentsInTlvSize += tlvTwoHeadersSize + std::get<1>(record);
    }

    TlvWriteT(outVector, "GNAI");
    TlvWriteL(outVector, tlvHeaderSize + allGNAMComponentsInTlvSize);
    TlvWriteT(outVector, "GNAM");
    TlvWriteL(outVector, allGNAMComponentsInTlvSize);

    // IN, OUT, SCRA of 20 Bytes (recordWithSizeInTlvSize) each
    for (const auto& record : recordsWithSizeOnly) {
        TlvWriteT(outVector, record.first);
        TlvWriteL(outVector, sizeInTlvSize);
        TlvWriteT(outVector, "SIZE");
        TlvWriteL(outVector, lInTlvSize);
        TlvWriteL(outVector, record.second);
    }
    // LDA, RORW, DBG of variable sizes
    for (const auto& record : recordsWithDataOnly) {
        TlvWriteT(outVector, std::get<0>(record));
        TlvWriteL(outVector, std::get<1>(record) + tlvHeaderSize);
        TlvWriteT(outVector, "DATA");
        TlvWriteL(outVector, std::get<1>(record));
        outVector.insert(outVector.end(), std::get<2>(record), std::get<2>(record) + std::get<1>(record));
    }
    return outVector;
}

void DumpToFile(std::string fileName, const std::vector<char>& data)
{
    std::ofstream outStream(fileName, std::ofstream::binary | std::ofstream::out);
    outStream.write(data.data(), data.size());
}

TEST_F(TestSimpleModelTlv, exportTlvNoMmuApi2)
{
    enablePwl = true;
    minimizeRw = true;
    SetupGnaModel();
    const auto out = Gna3DumpNoMmuTlv();
    // DumpToFile(test_info_->name() + std::string(".tlv"), out);
    ExpectLdEqual(out, refLDFromTlvNoMmuApi2);
    if (!separateInputAndOutput)
    {
        gnamem_pinned_inputs = nullptr;
        gnamem_pinned_outputs = nullptr;
    }
}

TEST_F(TestSimpleModelTlv, exportTlvNoMmuApi2_TagMemory)
{
    enablePwl = true;
    minimizeRw = true;
    separateInputAndOutput = true;
    SetupGnaModel();
    const auto out = Gna3DumpNoMmuTlv();
    // DumpToFile(test_info_->name() + std::string(".tlv"), out);
    ExpectLdEqual(out, refLDFromTlvNoMmuApi2_TagMemory);
    if (!separateInputAndOutput)
    {
        gnamem_pinned_inputs = nullptr;
        gnamem_pinned_outputs = nullptr;
    }
}
