/*
 @copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-tlv-anna-writer.h
 @brief Gaussian and Neural Accelerator (GNA) 3.1 TLV Anna writer.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA3_API Gaussian and Neural Accelerator (GNA) 3.1 API.
 @{

 ******************************************************************************

 @addtogroup GNA2_MODEL_EXPORT_API Embedded Model Export for Anna.

 @{
 *****************************************************************************/

#ifndef __GNA2_TLV_ANNA_WRITER_H
#define __GNA2_TLV_ANNA_WRITER_H

#include "gna2-tlv-anna.h"

#include <cstddef>
#include <cstdint>

GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvImplComputePadSize(
    uint32_t currentOffset,
    uint32_t alignment,
    uint32_t minPadSize,
    uint32_t * outPadSize)
{
    GNA2_TLV_EXPECT_NOT_NULL(outPadSize);

    const uint32_t const4k = 4096;
    if (alignment == 0 || alignment > const4k || minPadSize > const4k)
    {
        return Gna2TlvStatusNotSupported;
    }
    uint32_t paddingNeeded = (alignment - (((currentOffset % alignment) + minPadSize) % alignment)) % alignment;
    if (paddingNeeded > 0 && paddingNeeded < minPadSize)
    {
        paddingNeeded += alignment;
    }
    *outPadSize = paddingNeeded;
    return Gna2TlvStatusSuccess;
}

GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvImplPad(char* buf, uint32_t paddingTlvLength)
{
    GNA2_TLV_EXPECT_NOT_NULL(buf);

    const auto elem = reinterpret_cast<Gna2TlvRecord*>(buf);
    elem->type = Gna2TlvTypePadding;
    elem->length = paddingTlvLength;
    buf += GNA2_TLV_EMPTY_RECORD_SIZE;

    for (uint32_t i = 0; i < paddingTlvLength; i++)
    {
        buf[0] = '*';
        buf++;
    }
    return Gna2TlvStatusSuccess;
}

GNA2_TLV_LINKAGE void Gna2TlvImplCopy(char*& outBuf, const char * src, uint32_t srcLength)
{
    while(srcLength--)
    {
        *outBuf++ = *src++;
    }
}

GNA2_TLV_LINKAGE void Gna2TlvImplWriteTypeLength(char*&buffer, Gna2TlvType type, Gna2TlvLength length)
{
    const auto element = reinterpret_cast<Gna2TlvRecord*>(buffer);
    element->type = type;
    element->length = length;
    buffer += GNA2_TLV_EMPTY_RECORD_SIZE;
}

GNA2_TLV_LINKAGE void Gna2TlvImplWrite4BSize(char*&buffer, Gna2TlvType type, uint32_t sizeAsValue)
{
    Gna2TlvImplWriteTypeLength(buffer, type, sizeof(uint32_t));
    *reinterpret_cast<uint32_t*>(buffer) = sizeAsValue;
    buffer += sizeof(uint32_t);
}

GNA2_TLV_LINKAGE Gna2TlvStatus Gna2ExportAnnaTlv(
    Gna2TlvAllocator userAllocatorIn,   // [in] allocator from user (if called outTlv and outTlvSize are also written)
    char ** outTlv,                     // [out] address of serialized TLV
    uint32_t * outTlvSize,              // [out] size of serialized TLV
    const char* lda,                    // [in] [required not null] layer descriptor component, as exported with Gna2ModelExport() and Gna2ModelExportComponentLayerDescriptors
    uint32_t ldaSize,                   // [in] layer descriptor size in bytes, as exported with Gna2ModelExport() and Gna2ModelExportComponentLayerDescriptors
    const char* ro,                     // [opt] read only model fragment
    uint32_t roSize,                    // read only model part size, must be 0 if ro is NULL
    const char* state,                  // [opt] state model fragment
    uint32_t stateSize,                 // state model fragment size, must be 0 if state is NULL
    uint32_t scratchSize,               // scratch model fragment size
    uint32_t externalInputSize,         // external input buffer size of model
    uint32_t externalOutputSize,        // external output buffer size of model
    const char* gnaLibraryVersion,      // [opt] GNA library's version c-string obtained with Gna2GetLibraryVersion(), if the model is exported using GNA library, can be NULL otherwise
    const char* userData,               // [opt] optional user data
    uint32_t userDataSize               // user data size, must be 0 if userData is NULL
)
{
    uint32_t gnaLibraryVersionLength = 0;
    const uint32_t maxGnaLibraryVersionLength = 1024;

    if (gnaLibraryVersion != NULL)
    {
        while(gnaLibraryVersion[gnaLibraryVersionLength] != '\0' && gnaLibraryVersionLength < maxGnaLibraryVersionLength)
        {
            gnaLibraryVersionLength++;
        }
        gnaLibraryVersionLength++;  // count the terminating null-character
        if(gnaLibraryVersionLength > maxGnaLibraryVersionLength)
        {
            return Gna2TlvStatusLengthTooBig;
        }
    }

    GNA2_TLV_EXPECT_NOT_NULL(lda);
    GNA2_TLV_EXPECT_NOT_NULL(userAllocatorIn);
    GNA2_TLV_EXPECT_NOT_NULL(outTlv);
    GNA2_TLV_EXPECT_NOT_NULL(outTlvSize);

    GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(stateSize, state);
    GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(roSize, ro);
    GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(userDataSize, userData);

    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(ldaSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(roSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(stateSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(scratchSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(externalInputSize);
    GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(userDataSize);

    const uint32_t numberOfRecordsBeforeLda = 5;
    uint32_t outPadSizeLda = 0;
    const uint32_t sizeOfRecordsBeforeLda = numberOfRecordsBeforeLda * (GNA2_TLV_EMPTY_RECORD_SIZE + sizeof(uint32_t));
    Gna2TlvStatus status = Gna2TlvImplComputePadSize(sizeOfRecordsBeforeLda, GNA2_TLV_ANNA_REQUIRED_ALIGNEMENT, GNA2_TLV_EMPTY_RECORD_SIZE, &outPadSizeLda);
    if (status != Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    const uint32_t sizeOfLdaRecord = ldaSize + roSize + GNA2_TLV_EMPTY_RECORD_SIZE;
    const uint32_t sizeOfRecordsBeforeState = sizeOfLdaRecord + outPadSizeLda + sizeOfRecordsBeforeLda;
    uint32_t outPadSizeState = 0;
    status = Gna2TlvImplComputePadSize(sizeOfRecordsBeforeState, GNA2_TLV_ANNA_REQUIRED_ALIGNEMENT, GNA2_TLV_EMPTY_RECORD_SIZE, &outPadSizeState);
    if (status != Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusOutOfBuffer;
    }

    const uint32_t totalSizeRequired = sizeOfRecordsBeforeState + outPadSizeState + stateSize + userDataSize +
        3 * (GNA2_TLV_EMPTY_RECORD_SIZE) + gnaLibraryVersionLength;

    *outTlv = (char*)userAllocatorIn(totalSizeRequired);
    *outTlvSize = totalSizeRequired;

    if(*outTlv == NULL)
    {
        return Gna2TlvStatusUserAllocatorError;
    }
    char* curOutBuffer = *outTlv;

    Gna2TlvImplWrite4BSize(curOutBuffer, Gna2TlvTypeTlvVersion, GNA2_TLV_VERSION);
    Gna2TlvImplWrite4BSize(curOutBuffer, Gna2TlvTypeLayerDescriptorArraySize, ldaSize);
    Gna2TlvImplWrite4BSize(curOutBuffer, Gna2TlvTypeScratchSize, scratchSize);
    Gna2TlvImplWrite4BSize(curOutBuffer, Gna2TlvTypeExternalInputBufferSize, externalInputSize);
    Gna2TlvImplWrite4BSize(curOutBuffer, Gna2TlvTypeExternalOutputBufferSize, externalOutputSize);

    if(outPadSizeLda != 0)
    {
        const Gna2TlvStatus status = Gna2TlvImplPad(curOutBuffer, outPadSizeLda - GNA2_TLV_EMPTY_RECORD_SIZE);
        if(status != Gna2TlvStatusSuccess)
        {
            return Gna2TlvStatusOutOfBuffer;
        }
        curOutBuffer += outPadSizeLda;
    }

    Gna2TlvImplWriteTypeLength(curOutBuffer, Gna2TlvTypeLayerDescriptorAndRoArrayData, ldaSize + roSize);
    Gna2TlvImplCopy(curOutBuffer, lda, ldaSize);
    Gna2TlvImplCopy(curOutBuffer, ro, roSize);

    if (outPadSizeState != 0)
    {
        Gna2TlvStatus status = Gna2TlvImplPad(curOutBuffer, outPadSizeState - GNA2_TLV_EMPTY_RECORD_SIZE);
        if (status != Gna2TlvStatusSuccess)
        {
            return Gna2TlvStatusOutOfBuffer;
        }
        curOutBuffer += outPadSizeState;
    }

    Gna2TlvImplWriteTypeLength(curOutBuffer, Gna2TlvTypeStateData, stateSize);
    Gna2TlvImplCopy(curOutBuffer, state, stateSize);

    Gna2TlvImplWriteTypeLength(curOutBuffer, Gna2TlvTypeGnaLibraryVersionString, gnaLibraryVersionLength);
    Gna2TlvImplCopy(curOutBuffer, gnaLibraryVersion, gnaLibraryVersionLength);

    Gna2TlvImplWriteTypeLength(curOutBuffer, Gna2TlvTypeUserData, userDataSize);
    Gna2TlvImplCopy(curOutBuffer, userData, userDataSize);

    if(curOutBuffer != *outTlv + totalSizeRequired)
    {
        return Gna2TlvStatusUnknownError;
    }

    return Gna2TlvStatusSuccess;
}

#endif // __GNA2_TLV_ANNA_WRITER_H

/**
 @}
 @}
 */
