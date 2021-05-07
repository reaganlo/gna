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
#include "TlvErrors.h"
#include "TlvRead.h"
#include "TlvWrite.h"

#include <stdbool.h>
#include <stdio.h>

static uint32_t nReadFrames = 0;
static struct TlvFrame* readFrames = NULL;
static uint32_t countFrames = 0;

static void TlvReadTypeAndLength(void* data, uint32_t* size,
    char* readType, uint32_t* readOffset, struct TlvFrame* currentFrame, enum TlvStatus* status)
{
    TlvCheckNotNull(data, status);
    TlvCheckNotNull(size, status);
    TlvCheckNotNull(readType, status);
    TlvCheckNotNull(readOffset, status);
    TlvCheckNotNull(currentFrame, status);

    if (*size < TLV_TYPE_AND_LENGTH_FIELDS_SIZE)
    {
        *status = TLV_ERROR_INVALID_SIZE;
    }
    if (*status != 0)
    {
        return;
    }

    memcpy_s(&currentFrame->type, sizeof(currentFrame->type), data,
        TLV_TYPE_ID_SIZE);
    memcpy_s(&currentFrame->length, sizeof(currentFrame->length),
        (uint8_t*)data + TLV_TYPE_ID_SIZE, TLV_LENGTH_SIZE);

    *size -= TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
    *readOffset = TLV_TYPE_AND_LENGTH_FIELDS_SIZE;

    memcpy_s(readType, TLV_TYPE_ID_SIZE, data, TLV_TYPE_ID_SIZE);
    readType[TLV_TYPE_ID_SIZE] = '\0';

    currentFrame->childrenNodes = NULL;
    currentFrame->numberOfChildrenNodes = 0;
    currentFrame->value = NULL;
    currentFrame->parentNode = NULL;
}

bool isRawType(const TlvTypeId* value)
{
    TlvExitOnNull(value);
    for (uint32_t i = 0; i < rawTypesListSize; ++i)
    {
        if (value->numberValue == currentRawTypesList[i].numberValue)
        {
            return true;
        }
    }
    return false;
}

enum TlvStatus TlvGetSize(void* data, uint32_t inputSize,
    uint32_t* returnedSize)
{
    enum TlvStatus status = TLV_SUCCESS;
    TlvCheckNotNull(data, &status);
    TlvCheckNotNull(returnedSize, &status);
    if (inputSize < (uint32_t)TLV_TYPE_AND_LENGTH_FIELDS_SIZE)
    {
        status = TLV_ERROR_INVALID_SIZE;
    }
    if (status != 0)
    {
        return status;
    }

    TlvTypeId readType[TLV_TYPE_ID_SIZE];
    uint32_t readLen = 0;
    uint32_t offset = 0;

    while (inputSize > 0)
    {
        uint8_t* src = (uint8_t*)data + offset;
        memcpy_s(readType, sizeof(readType), src, TLV_TYPE_ID_SIZE);

        src = (uint8_t*)data + TLV_TYPE_ID_SIZE + offset;
        memcpy_s(&readLen, sizeof(uint32_t), src, TLV_LENGTH_SIZE);

        if (isRawType(readType))
        {
            inputSize -= readLen;
            offset += readLen;
        }
        inputSize -= TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
        offset += TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
        countFrames += 1;
    }
    *returnedSize = countFrames * (uint32_t)sizeof(struct TlvFrame);
    return status;
}

void UpdateChildren(struct TlvFrame* frame)
{
    TlvExitOnNull(frame);
    for (uint32_t frameIndex = 0; frameIndex < nReadFrames; ++frameIndex)
    {
        frame = readFrames + frameIndex;
        frame->childrenNodes = NULL;
    }

    for (uint32_t i = 1; i < nReadFrames; ++i)
    {
        frame = readFrames + i;
        if (frame->parentNode->childrenNodes == NULL)
        {
            frame->parentNode->childrenNodes = frame;
        }
    }
}

bool ifNodeFull(struct TlvFrame* parent)
{
    TlvExitOnNull(parent);
    uint32_t countSize = 0;
    for (uint32_t i = 0; i < parent->numberOfChildrenNodes; ++i)
    {
        countSize += TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
        countSize += parent->childrenNodes[i].length;
    }
    return countSize == parent->length ? true: false;
}

static int insertAndShiftFrames(struct TlvFrame* parentNode, uint32_t offset)
{
    TlvExitOnNull(parentNode);
    offset++;
    struct TlvFrame * insertFrame = readFrames + offset;
    while (insertFrame->parentNode == parentNode)
    {
        offset++;
        insertFrame = readFrames + offset;
    }

    if ((nReadFrames - offset) > 0)
    {
        const uint32_t allocSize = (nReadFrames - offset) * (uint32_t)sizeof(struct TlvFrame);
        void* tmpMem = malloc(allocSize);
        if (tmpMem == NULL)
        {
            printf("Tlv: cannot alloc tmp memory.\n");
            return TLV_ERROR_MEMORY_ALLOC;
        }

        memcpy_s(tmpMem, allocSize, insertFrame, allocSize);
        memcpy_s(insertFrame, sizeof(struct TlvFrame), readFrames + nReadFrames,
            sizeof(struct TlvFrame));
        memcpy_s(insertFrame + 1, allocSize, tmpMem, allocSize);

        free(tmpMem);
    }
    insertFrame->parentNode = parentNode;
    return 0;
}

void updateChildrenNodes()
{
    TlvExitOnNull(readFrames);
    struct TlvFrame* frame;
    for (uint32_t i = 0; i < nReadFrames; ++i)
    {
        frame = readFrames + i;
        frame->childrenNodes = NULL;
    }
    for (uint32_t i = 1; i <= nReadFrames; ++i)
    {
        frame = readFrames + i;
        if (frame->parentNode != NULL)
        {
            if (frame->parentNode->childrenNodes == NULL)
            {
                frame->parentNode->childrenNodes = frame;
            }
        }
    }
}

void TlvAddChildToParentFrame(struct TlvFrame* currentFrame)
{
    TlvExitOnNull(readFrames);
    uint32_t parentOffset = nReadFrames - 1;
    struct TlvFrame* parentNode = readFrames + parentOffset;
    bool shift = false;

    while (true)
    {
        if (ifNodeFull(parentNode) == false && !isRawType(&parentNode->type))
        {
            parentNode->numberOfChildrenNodes += 1;
            if (parentNode->childrenNodes == NULL)
            {
                parentNode->childrenNodes = currentFrame;
                currentFrame->parentNode = parentNode;
            }

            if (shift)
            {

                insertAndShiftFrames(parentNode, parentOffset);
                updateChildrenNodes();
            }
            break;
        }
        shift = true;
        parentOffset -= 1;
        parentNode = readFrames + parentOffset;
    }
}

static void copyData(const void* data, uint32_t* size, uint32_t* readOffset,
    struct TlvFrame* currentFrame)
{
    uint8_t* dataSrc = (uint8_t*)data + TLV_TYPE_ID_SIZE + TLV_LENGTH_SIZE;

    currentFrame->value = dataSrc;

    *readOffset += currentFrame->length;
    *size -= currentFrame->length;
}

static enum TlvStatus TlvReadFrame(void* data, uint32_t* size, uint32_t frameOffset,
    uint32_t* readOffset)
{
    enum TlvStatus status = TLV_SUCCESS;
    TlvCheckNotNull(readFrames, &status);
    struct TlvFrame* currentFrame = currentFrame = readFrames + nReadFrames;
    static TlvTypeId readType;
    TlvCheckNotNull(data, &status);
    TlvCheckIfNotZero(*size, &status);
    TlvReadTypeAndLength(data, size, readType.stringValue, readOffset,
        currentFrame, &status);

    if (isRawType(&readType) == true)
    {
        copyData(data, size, readOffset, currentFrame);
        currentFrame->childrenNodes = NULL;
    }

    if (nReadFrames > 0)
    {
        TlvAddChildToParentFrame(currentFrame);
    }

    nReadFrames++;

    if (*size > 0)
    {
        frameOffset += TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
        status = TlvReadFrame((uint8_t*)data + *readOffset, size, frameOffset,
            readOffset);
    }
    return status;
}

enum TlvStatus TlvDecode(void* data, uint32_t size, struct TlvFrame* memory,
    uint32_t* numberOfReadFrames)
{
    TlvExitOnNull(data);
    TlvExitOnNull(memory);
    TlvExitOnNull(numberOfReadFrames);
    nReadFrames = 0;
    readFrames = memory;
    const uint32_t frameOffset = 0;
    uint32_t readOffset = 0;
    const enum TlvStatus status = TlvReadFrame(data, &size, frameOffset, &readOffset);
    *numberOfReadFrames = nReadFrames;
    return status;
}

enum TlvStatus TlvLoadOwnRawList(const TlvTypeId ownRawTypeList[],
    uint32_t numberOfRawElements)
{
    currentRawTypesList = &ownRawTypeList[0];
    rawTypesListSize = numberOfRawElements;
    return TLV_SUCCESS;
}
