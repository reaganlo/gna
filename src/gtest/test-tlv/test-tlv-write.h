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

#pragma once

#include "TlvWrite.h"

#include "gtest/gtest.h"

static const TlvTypeId GNAI =  { 'G', 'N', 'A', 'I'};
static const TlvTypeId GNAM = { 'G', 'N', 'A', 'M' };
static const TlvTypeId IN = { 'I', 'N'};
static const TlvTypeId OUT = { 'O', 'U', 'T' };
static const TlvTypeId SCRA = { 'S', 'C', 'R', 'A' };
static const TlvTypeId LDT = { 'L', 'D', 'T' };
static const TlvTypeId RORW = { 'R', 'O', 'R', 'W' };
static const TlvTypeId DBG = { 'D', 'B', 'G' };
static const TlvTypeId SIZE = { 'S', 'I', 'Z', 'E' };
static const TlvTypeId DATA_TYPE = { 'D', 'A', 'T', 'A' };

class RecordInitTest : public ::testing::Test
{
public:
    uint32_t id = 0;
    const TlvTypeId TYPE = { {'G', 'N', 'A', 'I'} };
    void SetUp() override
    {
        id = 0;
    }
};

class RecordInitRawTest : public ::testing::Test
{
public:
    uint32_t id = 0;
    const TlvTypeId TYPE = { { 'G', 'N', 'A', 'I' } };
    uint32_t length = 5;
    void* data = NULL;

    void SetUp() override
    {
        id = 0;
        data = malloc(length);
    }

    void TearDown() override
    {
        free(data);
    }
};

class RecordAddTest : public ::testing::Test
{
public:
    uint32_t parent = 0;
    uint32_t child = 0;
    const TlvTypeId TYPE = { { 'G', 'N', 'A', 'I' } };
    void SetUp() override
    {
        parent = 0;
        child = 1;
    }
    void TearDown() override
    {
        TlvRecordsRelease(0);
    }

};

class RecordSerializeTest : public ::testing::Test
{
public:
    uint32_t id1, id2, id3, id4, id5, id6, id7, id8, id9, id10, id11, id12, id13, id14;
    void* mem = NULL;
    uint32_t dataSize;

    static const uint32_t rawSize = 5;
    static const uint32_t rawSize2 = 1;
    const char DATA[rawSize + 1] = "CHECK";
    const char DATA2[rawSize + 1] = "D";

    uint8_t serializeResult[5][142] = {
        { 71, 78, 65, 73, 8, 0, 0, 0, 71, 78, 65, 77, 0, 0, 0, 0 },
        { 71, 78, 65, 73, 13, 0, 0, 0, 73, 78, 0, 0, 5, 0, 0, 0, 67, 72, 69, 67, 75 },
        { 71, 78, 65, 73, 22, 0, 0, 0, 73, 78, 0, 0, 5, 0, 0, 0, 67, 72, 69, 67, 75,
          79, 85, 84, 0, 1, 0, 0, 0, 68 },
        { 71, 78, 65, 73, 30, 0, 0, 0, 73, 78, 0, 0, 5, 0, 0, 0, 67, 72, 69, 67, 75,
          79, 85, 84, 0, 9, 0, 0, 0, 83, 73, 90, 69, 1, 0, 0, 0, 68 },
        {
            71, 78, 65, 73, 134,  0,  0,  0, 71, 78,
            65, 77, 126,  0,  0,  0, 73, 78,  0,  0,
            13,  0,  0,  0, 83, 73, 90, 69,  5,  0,
            0,  0, 67, 72, 69, 67, 75, 79, 85, 84,
            0, 13,  0,  0,  0, 83, 73, 90, 69,  5,
            0,  0,  0, 67, 72, 69, 67, 75, 83, 67,
            82, 65, 13,  0,  0,  0, 83, 73, 90, 69,
            5,  0,  0,  0, 67, 72, 69, 67, 75, 76,
            68, 84,  0, 13,  0,  0,  0, 68, 65, 84,
            65,  5,  0,  0,  0, 67, 72, 69, 67, 75,
            82, 79, 82, 87, 13,  0,  0,  0, 68, 65,
            84, 65,  5,  0,  0,  0, 67, 72, 69, 67,
            75, 68, 66, 71,  0, 13,  0,  0,  0, 68,
            65, 84, 65,  5,  0,  0,  0, 67, 72, 69,
            67, 75}
    };
};

class RecordRelease : public ::testing::Test
{
public:
    uint32_t id1, id2, id3;

    void TearDown() override
    {
        TlvRecordsRelease(0);
    }
};
