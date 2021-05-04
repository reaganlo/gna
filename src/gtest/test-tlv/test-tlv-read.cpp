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
#include "TlvApi.h"
#include "TlvCommon.h"

#include "gtest/gtest.h"

#include <iostream>
#include <fstream>

uint32_t id1, id2, id3, id4, id5;
const TlvTypeId MAIN_STRING = { 'M', 'A', 'I', 'N' };
const TlvTypeId OWN = { 'O', 'W', 'N' };
const TlvTypeId TYPE = { 'T', 'Y', 'P', 'E' };
const TlvTypeId SUB = { 'S', 'U', 'B' };
const TlvTypeId SIZE_STRING = { 'S', 'I', 'Z', 'E' };
const TlvTypeId DATA_STING = { 'D', 'A', 'T', 'A' };

TEST(TlvRead, first)
{
    const uint8_t IN[] = { 0x49, 0x4e };
    const uint8_t TRY_DATA[] = { 0x54, 0x72, 0x79 };
    const uint8_t MEAN[] = { 0x6d, 0x65, 0x61, 0x6e };

    TlvRecordInit(MAIN_STRING, &id1);
    TlvRecordInit(SUB, &id2);
    TlvRecordInitRaw(SIZE_STRING, 2, &IN, &id3);
    TlvRecordInitRaw(DATA_STING, 4, &MEAN, &id4);
    TlvRecordInitRaw(DATA_STING, 3, &TRY_DATA, &id5);

    TlvRecordAdd(id1, id2);
    TlvRecordAdd(id2, id3);
    TlvRecordAdd(id2, id4);
    TlvRecordAdd(id2, id5);

    auto const expectedSize = 49;

    uint32_t size;
    enum TlvStatus status = TlvRecordGetSize(id1, &size);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(size, expectedSize);

    auto memory = malloc(size);
    TlvSerialize(id1, memory, size);

    //read
    uint32_t readSize;
    uint32_t numberOfReadFrames;

    TlvGetSize(memory, size, &readSize);

    TlvFrame* frames = (TlvFrame*)malloc(readSize);
    TlvDecode(memory, size, frames, &numberOfReadFrames);

    TlvFrame* second = frames + 1;
    TlvFrame* last = frames + 4;

    EXPECT_EQ(&second->childrenNodes[2], last);

    free(frames);
    free(memory);
}

TEST(TlvRead, second)
{
    const uint8_t DATA[] = { 0x66, 0x72, 0x61, 0x6d, 0x65, 0x73};
    const uint8_t D = 'd';
    const uint8_t OR_STRING [] = { 0x6f, 0x72 };

    TlvRecordInit(MAIN_STRING, &id1);
    TlvRecordInit(SUB, &id2);
    TlvRecordInitRaw(SIZE_STRING, 6, &DATA, &id3);
    TlvRecordInitRaw(SIZE_STRING, 1, &D, &id4);
    TlvRecordInitRaw(DATA_STING, 2, &OR_STRING, &id5);

    TlvRecordAdd(id1, id3);
    TlvRecordAdd(id1, id2);
    TlvRecordAdd(id2, id4);
    TlvRecordAdd(id2, id5);

    auto const expectedSize = 49;
    uint32_t size;
    enum TlvStatus status = TlvRecordGetSize(id1, &size);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(size, expectedSize);

    auto memory = malloc(size);
    TlvSerialize(id1, memory, size);

    //read
    uint32_t readSize, numberOfReadFrames;
    TlvGetSize(memory, size, &readSize);

    TlvFrame* frames = (TlvFrame*)malloc(readSize);
    TlvDecode(memory, size, frames, &numberOfReadFrames);

    TlvFrame* third = frames + 2;
    TlvFrame* last = frames + 4;

    EXPECT_EQ(&third->childrenNodes[1], last);

    free(frames);
    free(memory);
}

TEST(TlvLoadOwnRawListTest, first)
{
    const TlvTypeId MY_TYPE_LIST[] =
    {
        'O', 'W', 'N', '\0',
        'T', 'Y', 'P', 'E'
    };
    const char OWN [4]= "own";
    const char MY_TYPE [7] = "myType";
    TlvRecordInit(MAIN_STRING, &id1);
    TlvRecordInitRaw(MY_TYPE_LIST[0], 3, &OWN, &id2);
    TlvRecordInitRaw(MY_TYPE_LIST[1], 6, &MY_TYPE, &id3);
    TlvRecordAdd(id1, id2);
    TlvRecordAdd(id1, id3);

    auto const expectedSize = 33;
    uint32_t size;
    enum TlvStatus status = TlvRecordGetSize(id1, &size);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(size, expectedSize);

    auto memory = malloc(size);
    TlvSerialize(id1, memory, size);

    TlvLoadOwnRawList(MY_TYPE_LIST, 2);

    uint32_t readSize, numberOfReadFrames;
    TlvGetSize(memory, size, &readSize);

    TlvFrame* frames = (TlvFrame*)malloc(readSize);
    TlvDecode(memory, size, frames, &numberOfReadFrames);

    free(frames);
    free(memory);
}
