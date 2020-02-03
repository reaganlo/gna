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

#include "test-tlv-write.h"
#include "TlvCommon.h"

#include "TlvErrors.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdlib>
#include <stdio.h>
#include <string>

TEST_F(RecordInitTest, invalidIdPtr)
{
    TlvLibraryInit();
    EXPECT_EQ(TlvRecordInit(TYPE, NULL), TLV_ERROR_DATA_NULL);
}

TEST_F(RecordInitTest, validCheck)
{
    for (auto i = 0; i < 10; ++i)
    {
        enum TlvStatus status = TlvRecordInit(TYPE, &id);
        EXPECT_EQ(status, 0);
        EXPECT_EQ(id, i);

        TlvFrame* frame =  TlvGetRecord(id);

        char readType[TLV_TYPE_ID_SIZE];
        memcpy_s(readType, sizeof(readType), &frame->type, TLV_TYPE_ID_SIZE);

        EXPECT_EQ(memcmp(TYPE.stringValue, readType, TLV_TYPE_ID_SIZE), 0);

        uint32_t size;
        EXPECT_EQ(TlvRecordGetSize(id, &size), 0);
        EXPECT_EQ(size, TLV_TYPE_AND_LENGTH_FIELDS_SIZE);
    }
    for (auto i = 0; i < 10; ++i)
    {
        TlvRecordsRelease(i);
    }
}

TEST_F(RecordInitRawTest, invalidValue)
{
    EXPECT_EQ(TlvRecordInitRaw(TYPE, length, NULL, &id),
        TLV_ERROR_DATA_NULL);
}

TEST_F(RecordInitRawTest, invalidIdPtr)
{
    EXPECT_EQ(TlvRecordInitRaw(TYPE, length, data, NULL),
        TLV_ERROR_DATA_NULL);
}

TEST_F(RecordInitRawTest, valid)
{
    for (auto i = 0; i < 10; ++i)
    {
        const std::string DATA = "CHECK";

        EXPECT_EQ(TlvRecordInitRaw(TYPE, length, &DATA, &id), 0);
        EXPECT_EQ(id, i);

        TlvFrame* frame = TlvGetRecord(id);
        char readType[TLV_TYPE_ID_SIZE];

        memcpy_s(readType, sizeof(readType), &frame->type, TLV_TYPE_ID_SIZE);

        EXPECT_EQ(memcmp(TYPE.stringValue, readType, TLV_TYPE_ID_SIZE), 0);

        auto expectedSize = TLV_TYPE_AND_LENGTH_FIELDS_SIZE + length;
        uint32_t size;
        EXPECT_EQ(TlvRecordGetSize(id, &size), 0);
        EXPECT_EQ(size, expectedSize);
    }
    for (auto i = 0; i < 10; ++i)
    {
        TlvRecordsRelease(i);
    }
}

TEST_F(RecordAddTest, notEnoughAvaiableFrames)
{
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(TlvRecordAdd(parent, child),
            TLV_ERROR_ARGS_OUT_OF_RANGE);
    }
}

TEST_F(RecordAddTest, idsGreaterThanMax)
{
    uint32_t someValue = 2;
    parent = someValue + 1;

    EXPECT_EQ(TlvRecordAdd(parent, child), TLV_ERROR_ARGS_OUT_OF_RANGE);

    parent = someValue;
    child = someValue + 1;
    EXPECT_EQ(TlvRecordAdd(parent, child), TLV_ERROR_ARGS_OUT_OF_RANGE);
}

TEST_F(RecordAddTest, parentSameAdChild)
{
    EXPECT_EQ(TlvRecordAdd(parent, parent), TLV_ERROR_NODES_THE_SAME);
    EXPECT_EQ(TlvRecordAdd(child, child), TLV_ERROR_NODES_THE_SAME);
}


TEST_F(RecordSerializeTest, dataPtrNull)
{
    dataSize = 10;
    EXPECT_EQ(TlvSerialize(0, NULL, dataSize), TLV_ERROR_DATA_NULL);
}

TEST_F(RecordSerializeTest, frameNotExist)
{
    dataSize = 10;
    mem = malloc(dataSize);
    EXPECT_EQ(TlvSerialize(0, mem, dataSize), TLV_ERROR_ARGS_OUT_OF_RANGE);
    free(mem);
}

TEST_F(RecordSerializeTest, noAvaiableMemory)
{
    mem = malloc(10);
    EXPECT_EQ(TlvSerialize(0, mem, 0), TLV_ERROR_INVALID_SIZE);
    free(mem);
}

TEST_F(RecordSerializeTest, 1Parent1Child)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInit(GNAM, &id2);

    TlvRecordAdd(id1, id2);

    uint32_t dataSize;
    TlvRecordGetSize(id1, &dataSize);
    mem = malloc(dataSize);
    enum TlvStatus status = TlvSerialize(id1, mem, dataSize);

    EXPECT_EQ(status, 0);
    EXPECT_EQ(memcmp(mem, &serializeResult[0][0], 16), 0);

    free(mem);
    TlvRecordsRelease(id1);
}

TEST_F(RecordSerializeTest, 1Parent1ChildMemoryOverrun)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInit(GNAM, &id2);

    TlvRecordAdd(id1, id2);

    uint32_t dataSize;
    TlvRecordGetSize(id1, &dataSize);
    mem = malloc(dataSize);
    enum TlvStatus status = TlvSerialize(id1, mem, dataSize - 15);

    EXPECT_EQ(status, TLV_ERROR_MEMORY_OVERUN);

    free(mem);
    TlvRecordsRelease(id1);
}

TEST_F(RecordSerializeTest, 1Parent1ChildRaw)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInitRaw(IN, rawSize, DATA, &id2);

    TlvRecordAdd(id1, id2);

    uint32_t dataSize;
    TlvRecordGetSize(id1, &dataSize);
    mem = malloc(dataSize);

    int status = TlvSerialize(id1, mem, dataSize);

    EXPECT_EQ(status, 0);
    EXPECT_EQ(memcmp(mem, &serializeResult[1][0], 21), 0);

    free(mem);
    TlvRecordsRelease(id1);
}

TEST_F(RecordSerializeTest, 1Parent2ChildrendRaw)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInitRaw(IN, rawSize, DATA, &id2);
    TlvRecordInitRaw(OUT, rawSize2, DATA2, &id3);

    TlvRecordAdd(id1, id2);
    TlvRecordAdd(id1, id3);

    uint32_t dataSize;
    TlvRecordGetSize(id1, &dataSize);
    mem = malloc(dataSize);

    int status = TlvSerialize(id1, mem, dataSize);

    EXPECT_EQ(status, 0);
    EXPECT_EQ(memcmp(mem, &serializeResult[2][0], 30), 0);

    free(mem);
    TlvRecordsRelease(id1);
}

TEST_F(RecordSerializeTest, 1Parent2ChildrenMixed)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInitRaw(IN, rawSize, DATA, &id2);
    TlvRecordInit(OUT, &id3);
    TlvRecordInitRaw(SIZE, rawSize2, DATA2, &id4);

    TlvRecordAdd(id1, id2);
    TlvRecordAdd(id1, id3);
    TlvRecordAdd(id3, id4);

    uint32_t dataSize;
    TlvRecordGetSize(id1, &dataSize);
    mem = malloc(dataSize);

    enum TlvStatus status = TlvSerialize(id1, mem, dataSize);

    EXPECT_EQ(status, 0);
    EXPECT_EQ(memcmp(mem, &serializeResult[3][0], 38), 0);

    free(mem);
    TlvRecordsRelease(id1);
}

TEST_F(RecordSerializeTest, 1Parent2ChildrenMixed2)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInit(GNAM, &id14);
    TlvRecordInit(IN, &id12);
    TlvRecordInit(OUT, &id9);
    TlvRecordInit(SCRA, &id10);
    TlvRecordInit(LDT, &id6);
    TlvRecordInit(RORW, &id7);
    TlvRecordInit(DBG, &id8);

    TlvRecordInitRaw(SIZE, rawSize, DATA, &id3);
    TlvRecordInitRaw(SIZE, rawSize, DATA, &id4);
    TlvRecordInitRaw(SIZE, rawSize, DATA, &id11);

    TlvRecordInitRaw(DATA_TYPE, rawSize, DATA, &id2);
    TlvRecordInitRaw(DATA_TYPE, rawSize, DATA, &id5);
    TlvRecordInitRaw(DATA_TYPE, rawSize, DATA, &id13);

    TlvRecordAdd(id1, id14);
    TlvRecordAdd(id14, id12);
    TlvRecordAdd(id14, id9);
    TlvRecordAdd(id14, id10);
    TlvRecordAdd(id14, id6);
    TlvRecordAdd(id14, id7);
    TlvRecordAdd(id14, id8);

    TlvRecordAdd(id12, id3);
    TlvRecordAdd(id9, id4);
    TlvRecordAdd(id10, id11);

    TlvRecordAdd(id6, id2);
    TlvRecordAdd(id7, id5);
    TlvRecordAdd(id8, id13);

    uint32_t dataSize;
    TlvRecordGetSize(id1, &dataSize);
    mem = malloc(dataSize);

    int status = TlvSerialize(id1, mem, dataSize);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(memcmp(mem, &serializeResult[4][0], 142), 0);

    free(mem);
    TlvRecordsRelease(id1);
}

TEST_F(RecordRelease, allocTwiceAtTheBegging)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInit(GNAM, &id2);
    TlvRecordsRelease(id1);
    TlvRecordInit(SCRA, &id1);

    TlvFrame* frame = TlvGetRecord(id1);
    char readType[TLV_TYPE_ID_SIZE];

    memcpy_s(readType, sizeof(readType), &frame->type, TLV_TYPE_ID_SIZE);

    EXPECT_EQ(memcmp(SCRA.stringValue, readType, TLV_TYPE_ID_SIZE), 0);

    auto expectedSize = TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
    TlvRecordsRelease(id1);
    TlvRecordsRelease(id2);
}

TEST_F(RecordRelease, allocTwiceInTheMiddle)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInit(GNAM, &id2);
    TlvRecordInit(IN, &id3);
    TlvRecordsRelease(id2);
    TlvRecordInit(SCRA, &id2);

    TlvFrame* frame = TlvGetRecord(id2);
    char readType[TLV_TYPE_ID_SIZE];

    memcpy_s(readType, sizeof(readType), &frame->type, TLV_TYPE_ID_SIZE);

    EXPECT_EQ(memcmp(SCRA.stringValue, readType, TLV_TYPE_ID_SIZE), 0);

    auto expectedSize = TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
    TlvRecordsRelease(id1);
    TlvRecordsRelease(id2);
    TlvRecordsRelease(id3);
}

TEST_F(RecordRelease, allocTwiceAtTheEnd)
{
    TlvRecordInit(GNAI, &id1);
    TlvRecordInit(GNAM, &id2);
    TlvRecordInit(IN, &id3);
    TlvRecordsRelease(id3);
    TlvRecordInit(SCRA, &id3);

    TlvFrame* frame = TlvGetRecord(id3);
    char readType[TLV_TYPE_ID_SIZE];

    memcpy_s(readType, sizeof(readType), &frame->type, TLV_TYPE_ID_SIZE);

    EXPECT_EQ(memcmp(SCRA.stringValue, readType, TLV_TYPE_ID_SIZE), 0);

    auto expectedSize = TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
    TlvRecordsRelease(id1);
    TlvRecordsRelease(id2);
    TlvRecordsRelease(id3);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    return RUN_ALL_TESTS();
}

