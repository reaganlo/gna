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
#include "TlvWrite.h"

#include "TlvErrors.h"

#include <string.h>
#include <errno.h>

static struct TlvFrame* tlvFrameList = NULL;

static uint32_t tlvFrameListSize = 0;

static LinkStruct* linkList = NULL;

static uint32_t linkListSize = 0;

inline void TlvExitOnNull(const void* ptr)
{
    if (ptr == NULL)
    {
        printf("Tlv: NULL pointer argument was provided.\n");
        exit(-1);
    }
}

inline void TlvCheckNotNull(const void* ptr, enum TlvStatus* status)
{
    TlvExitOnNull(status);
    if (ptr == NULL)
    {
        printf("Tlv: NULL pointer argument was provided.\n");
        *status = TLV_ERROR_DATA_NULL;
    }
}

inline void TlvCheckIfNotZero(const uint32_t number, enum TlvStatus* status)
{
    TlvExitOnNull(status);
    if (number == 0)
    {
        *status = TLV_ERROR_ZERO_LENGTH;
    }
}

void TlvCheckIfExceedsMax(const uint32_t id, enum TlvStatus* status)
{
    TlvExitOnNull(status);
    if (id >= tlvFrameListSize)
    {
        *status = TLV_ERROR_ARGS_OUT_OF_RANGE;
    }
}

void TlvLibraryInit()
{
    if (tlvFrameList)
    {
        free(tlvFrameList);
    }
    if (linkList)
    {
        free(linkList);
    }
    tlvFrameList = NULL;
    tlvFrameListSize = 0;
    linkList = NULL;
    linkListSize = 0;
}

static inline uint32_t checkIfEmptyFrame()
{
    struct TlvFrame* frame;
    for (uint32_t i = 0; i < tlvFrameListSize; ++i)
    {
        frame = tlvFrameList + i;
        if (frame->type.numberValue == 0)
        {
            return i;
        }
    }
    return UINT32_MAX;
}

static enum TlvStatus* determineCurrentRecord(struct TlvFrame** frame,
    uint32_t* id, enum TlvStatus* status)
{
    TlvCheckNotNull(id, status);
    if (*status != TLV_SUCCESS)
    {
        return status;
    }

    uint32_t emptyIndex = checkIfEmptyFrame();
    if (emptyIndex == UINT32_MAX)
    {
        *id = tlvFrameListSize;
        void* tmpTlvFrameList = realloc(tlvFrameList,
            (tlvFrameListSize + 1) * sizeof(struct TlvFrame));
        if (tmpTlvFrameList == NULL)
        {
            *status = TLV_ERROR_MEMORY_ALLOC;
            return status;
        }
        tlvFrameList = tmpTlvFrameList;

        *frame = tlvFrameList + tlvFrameListSize;
        tlvFrameListSize++;
    }
    else
    {
        *id = emptyIndex;
        *frame = tlvFrameList + emptyIndex;
    }
    return status;
}

enum TlvStatus TlvRecordInit(const TlvTypeId type, uint32_t* id)
{
    enum TlvStatus status = TLV_SUCCESS;

    TlvCheckNotNull(type.stringValue, &status);
    TlvCheckNotNull(id, &status);
    CHECK_ERROR(status);
    struct TlvFrame* frame = NULL;

    determineCurrentRecord(&frame, id, &status);
    CHECK_ERROR(status);

    memcpy_s(&frame->type, sizeof(frame->type), &type, TLV_TYPE_ID_SIZE);

    frame->length = 0;
    frame->numberOfChildrenNodes = 0;
    frame->childrenNodes = NULL;

    return status;
}

enum TlvStatus TlvRecordInitRaw(const TlvTypeId type, uint32_t length,
    const void* value, uint32_t* id)
{
    enum TlvStatus status = TLV_SUCCESS;

    TlvCheckNotNull(type.stringValue, &status);
    TlvCheckIfNotZero(length, &status);
    TlvCheckNotNull(value, &status);
    TlvCheckNotNull(id, &status);
    CHECK_ERROR(status);
    struct TlvFrame* frame;

    determineCurrentRecord(&frame, id, &status);
    CHECK_ERROR(status);

    memcpy_s(&frame->type, sizeof(frame->type), &type, TLV_TYPE_ID_SIZE);
    memcpy_s(&frame->length, sizeof(frame->length), &length, TLV_LENGTH_SIZE);

    frame->value = value;
    frame->numberOfChildrenNodes = 0;
    frame->childrenNodes = NULL;

    return status;
}

struct TlvFrame* TlvGetRecord(uint32_t id)
{
    return tlvFrameList + id;
}

enum TlvStatus TlvRecordAdd(uint32_t parentRecordId, uint32_t childRecordId)
{
    enum TlvStatus status = TLV_SUCCESS;
    if (parentRecordId >= tlvFrameListSize || childRecordId >= tlvFrameListSize)
    {
        status = TLV_ERROR_ARGS_OUT_OF_RANGE;
    }
    if (parentRecordId == childRecordId)
    {
        status = TLV_ERROR_NODES_THE_SAME;
    }
    CHECK_ERROR(status);
    linkListSize++;
    void* tmpAddList = realloc(linkList, linkListSize * sizeof(LinkStruct));
    if (tmpAddList == NULL)
    {
        status = TLV_ERROR_MEMORY_ALLOC;
    }
    linkList = tmpAddList;
    LinkStruct* currentLinkStruct = linkList + linkListSize - 1;
    currentLinkStruct->parentRecordId = parentRecordId;
    currentLinkStruct->childRecordId = childRecordId;
    return status;
}

int TlvAssign()
{
    enum TlvStatus status = TLV_SUCCESS;
    struct TlvFrame *parent, *child;
    LinkStruct* currentLink;

    for (uint32_t i = 0; i < linkListSize; ++i)
    {
        currentLink = linkList + i;
        parent = tlvFrameList + currentLink->parentRecordId;
        child = tlvFrameList + currentLink->childRecordId;
        parent->numberOfChildrenNodes++;
        child->parentNode = parent;
        parent->length += child->length + TLV_TYPE_AND_LENGTH_FIELDS_SIZE;

        if ((size_t)(child - parent) >
            (parent->numberOfChildrenNodes * sizeof(struct TlvFrame)))
        {
            size_t memoryBlockSize = sizeof(child - parent)
                - parent->numberOfChildrenNodes * sizeof(struct TlvFrame);
            void* tmpMemory = malloc(memoryBlockSize);
            if (tmpMemory == NULL)
            {
                status = TLV_ERROR_MEMORY_ALLOC;
            }
            memcpy_s(tmpMemory, memoryBlockSize,
                parent + parent->numberOfChildrenNodes, memoryBlockSize);
            memcpy_s(parent + parent->numberOfChildrenNodes,
                sizeof(struct TlvFrame), child, sizeof(struct TlvFrame));
            memcpy_s(parent + parent->numberOfChildrenNodes + 1,
                memoryBlockSize, tmpMemory, memoryBlockSize);
            if (parent->childrenNodes == NULL)
            {
                parent->childrenNodes = parent + 1;
            }
            free(tmpMemory);
        }
    }
    return status;
}

static void getSizeOfChildren(uint32_t id, uint32_t* recordSizeOut)
{
    TlvExitOnNull(recordSizeOut);
    LinkStruct* currentCheck;
    for (uint32_t i = 0; i < linkListSize; ++i)
    {
        currentCheck = linkList + i;
        if (currentCheck->parentRecordId == id)
        {
            *recordSizeOut += TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
            struct TlvFrame* checkChild = tlvFrameList
                + currentCheck->childRecordId;
            if (checkChild->length > 0)
            {
                *recordSizeOut += checkChild->length;
            }
            else
            {
                getSizeOfChildren(currentCheck->childRecordId, recordSizeOut);
            }
        }
    }
}

enum TlvStatus TlvRecordGetSize(uint32_t id, uint32_t* recordSizeOut)
{
    enum TlvStatus status = TLV_SUCCESS;
    TlvCheckIfExceedsMax(id, &status);
    TlvCheckNotNull(recordSizeOut, &status);
    CHECK_ERROR(status);

    *recordSizeOut = TLV_TYPE_AND_LENGTH_FIELDS_SIZE;
    struct TlvFrame* checkCurrentIsRaw = tlvFrameList + id;
    if (checkCurrentIsRaw->length > 0)
    {
        *recordSizeOut += checkCurrentIsRaw->length;
    }
    getSizeOfChildren(id, recordSizeOut);

    return status;
}

static enum TlvStatus RecordSerializeWriteFrame(uint32_t id, void* data,
    uint32_t* offset, const uint32_t dataSize)
{
    struct TlvFrame* frame = tlvFrameList + id;
    if (data == NULL)
    {
        return TLV_ERROR_DATA_NULL;
    }
    if (offset == NULL)
    {
        return TLV_ERROR_DATA_NULL;
    }
    memcpy_s((char*)data + *offset, TLV_TYPE_ID_SIZE, frame,
        TLV_TYPE_ID_SIZE);
    if ((*offset += TLV_TYPE_ID_SIZE) > dataSize)
    {
       return TLV_ERROR_MEMORY_OVERRUN;
    }

    if (frame->length > 0)
    {
        memcpy_s((char*)data + *offset, TLV_LENGTH_SIZE, &frame->length,
            sizeof(frame->length));
        if ((*offset += TLV_LENGTH_SIZE) > dataSize)
        {
            return TLV_ERROR_MEMORY_OVERRUN;
        }
        memcpy_s((char*)data + *offset, frame->length, frame->value,
            frame->length);

        if ((*offset += frame->length) > dataSize)
        {
            return TLV_ERROR_MEMORY_OVERRUN;
        }
    }
    else
    {
        uint32_t childrenSize = 0;
        getSizeOfChildren(id, &childrenSize);
        memcpy_s((char*)data + *offset, TLV_LENGTH_SIZE, &childrenSize,
            sizeof(childrenSize));
        if ((*offset += TLV_LENGTH_SIZE) > dataSize)
        {
            return TLV_ERROR_MEMORY_OVERRUN;
        }

        for (uint32_t i = 0; i < linkListSize; ++i)
        {
            LinkStruct* currentLink = linkList + i;
            if (currentLink->parentRecordId == id)
            {
                RecordSerializeWriteFrame(currentLink->childRecordId, data,
                    offset, dataSize);
            }
        }
    }
    return TLV_SUCCESS;
}

enum TlvStatus TlvSerialize(uint32_t id, void* data, const uint32_t dataSize)
{
    enum TlvStatus status = TLV_SUCCESS;
    TlvCheckIfExceedsMax(id, &status);
    TlvCheckNotNull(data, &status);

    if (dataSize == 0)
    {
        status = TLV_ERROR_INVALID_SIZE;
    }

    if (status != 0)
    {
        return status;
    }

    uint32_t offset = 0;
    status = RecordSerializeWriteFrame(id, data, &offset, dataSize);

    return status;
}

static bool findLastChildNode(uint32_t searchedId, uint32_t* idOfLastChild,
    uint32_t* linkIdToDelete)
{
    TlvExitOnNull(idOfLastChild);
    TlvExitOnNull(linkIdToDelete);
    for (uint32_t i = linkListSize - 1; i != UINT32_MAX; i--)
    {
        if (linkList[i].parentRecordId == searchedId)
        {
            *idOfLastChild = linkList[i].childRecordId;
            *linkIdToDelete = i;
            return true;
        }
    }
    return false;
}

static bool checkIfFrameEmpty(uint32_t id)
{
    struct TlvFrame* frame = tlvFrameList + id;
    if (memcmp(&frame->type, "\0\0\0\0", TLV_TYPE_ID_SIZE) == 0)
    {
        return true;
    }
    return false;
}

static enum TlvStatus releaseRecord(uint32_t id, enum TlvStatus* status)
{
    TlvExitOnNull(status);
    void* tmp;
    if (tlvFrameListSize == 1 && id == 0)
    {
        tlvFrameListSize = 0;
        free(tlvFrameList);
        tlvFrameList = NULL;
        return *status;
    }
    if (id == tlvFrameListSize - 1)
    {
        tlvFrameListSize--;
        tmp = realloc(tlvFrameList, tlvFrameListSize * sizeof(struct TlvFrame));
        if (tmp == NULL)
        {
            *status = TLV_ERROR_MEMORY_ALLOC;
            return *status;
        }
        tlvFrameList = tmp;

        uint32_t emptyFramesAtTheEnd = 0;
        uint32_t checkFrameId = tlvFrameListSize - 1;

        while (checkFrameId != UINT32_MAX && checkIfFrameEmpty(checkFrameId))
        {
            emptyFramesAtTheEnd++;
            checkFrameId--;
        }

        if (emptyFramesAtTheEnd > 0)
        {
            tlvFrameListSize -= emptyFramesAtTheEnd;
            if (tlvFrameListSize == 0)
            {
                free(tlvFrameList);
                tlvFrameList = NULL;
                return *status;
            }

            tmp = realloc(tlvFrameList,
                tlvFrameListSize * sizeof(struct TlvFrame));
            if (tmp == NULL)
            {
                *status = TLV_ERROR_MEMORY_ALLOC;
                return *status;
            }
            tlvFrameList = tmp;
        }
    }
    else
    {
        struct TlvFrame* frame = tlvFrameList + id;
        memset(&frame->type, 0, TLV_TYPE_ID_SIZE);
        frame->length = 0;
    }
    return *status;
}

static int releaseLinkRecord(uint32_t id, enum TlvStatus* status)
{
    TlvExitOnNull(status);
    void* tmp;
    if (linkListSize == 1 && id == 0)
    {
        linkListSize = 0;
        free(linkList);
        linkList = NULL;
        return *status;
    }

    if (id != linkListSize - 1)
    {
        uint32_t size = (linkListSize - id - 1) * (uint32_t)sizeof(LinkStruct);
        tmp = malloc(size);
        if (tmp == NULL)
        {
            *status = TLV_ERROR_MEMORY_ALLOC;
            return *status;
        }
        memcpy_s(tmp, size, linkList + id + 1, size);
        memcpy_s(linkList + id, size, tmp, size);
        free(tmp);
    }

    tmp = realloc(linkList, (linkListSize - 1) * sizeof(LinkStruct));
    if (tmp == NULL)
    {
        *status = TLV_ERROR_MEMORY_ALLOC;
        return *status;
    }
    linkList = tmp;
    linkListSize--;
    return *status;
}

static bool doesRecordContainChildren(uint32_t id)
{
    for (uint32_t i = 0; i < linkListSize; ++i)
    {
        if (linkList[i].parentRecordId == id)
        {
            return true;
        }
    }
    return false;
}

static enum TlvStatus deleteLinkIfChildForAFrame(uint32_t id)
{
    enum TlvStatus status = TLV_SUCCESS;
    LinkStruct* link;
    for (uint32_t i = 0; i < linkListSize; ++i)
    {
        link = linkList + i;
        if (link->childRecordId == id)
        {
            releaseLinkRecord(i, &status);
        }
    }
    return status;
}

enum TlvStatus TlvRecordsRelease(uint32_t id)
{
    enum TlvStatus status = TLV_SUCCESS;
    TlvCheckIfExceedsMax(id, &status);
    CHECK_ERROR(status);
    uint32_t currentSearch = id;
    uint32_t idOfLastChild = UINT32_MAX;
    uint32_t idLinkToRelease = UINT32_MAX;
    bool found;
    while (doesRecordContainChildren(id))
    {
        currentSearch = id;
        do
        {
            found = findLastChildNode(currentSearch, &idOfLastChild,
                &idLinkToRelease);
            currentSearch = idOfLastChild;
        } while (found);
        releaseRecord(idOfLastChild, &status);
        CHECK_ERROR(status);
        releaseLinkRecord(idLinkToRelease, &status);
        CHECK_ERROR(status);
    }
    deleteLinkIfChildForAFrame(id);
    CHECK_ERROR(status);

    return releaseRecord(id, &status);
}
