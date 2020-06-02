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

#include "gna2-capability-api.h"

#include <cstdint>
#include <string>

inline void Gna2TlvImplWriteType(char*&buffer, Gna2TlvType type)
{
    *((Gna2TlvType*)buffer) = type;
    buffer += 4;
}

inline void Gna2TlvImplWriteLength(char*&buffer, uint32_t length)
{
    *reinterpret_cast<uint32_t*>(buffer) = length;
    buffer += 4;
}

inline Gna2TlvStatus Gna2TlvImplComputePadSize(
    uint32_t currentOffset,
    uint32_t alignment,
    uint32_t minPadSize,
    uint32_t * outPadSize)
{
    const uint32_t const4k = 4096;
    if (outPadSize == NULL)
    {
        return Gna2TlvStatusNullNotAllowed;
    }
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

inline Gna2TlvStatus Gna2TlvImplPad(char* buf, uint32_t paddingTlvLength)
{
    if (buf == NULL)
    {
        return Gna2TlvStatusNullNotAllowed;
    }
    Gna2TlvImplWriteType(buf, TlvTypePadding);
    Gna2TlvImplWriteLength(buf, paddingTlvLength);

    for (uint32_t i = 0; i < paddingTlvLength; i++)
    {
        buf[0] = '*';
        buf++;
    }
    return Gna2TlvStatusSuccess;
}

inline void Gna2TlvImplCopy(char*& outBuf, const char * src, uint32_t srcLength)
{
    while(srcLength--)
    {
        *outBuf++ = *src++;
    }
}

void Gna2TlvImplWrite4BSize(char*&buffer, Gna2TlvType type, uint32_t sizeAsValue)
{
    Gna2TlvImplWriteType(buffer, type);
    Gna2TlvImplWriteLength(buffer, 4);
    Gna2TlvImplWriteLength(buffer, sizeAsValue);
}

#define GNA_TLV_EXPECT_NOT_NULL(ptr) {if((ptr) == NULL) return Gna2TlvStatusNullNotAllowed;}

inline Gna2TlvStatus Gna2ExportAnnaTlv(
    Gna2TlvAllocator userAllocatorIn,   // [in] allocator from user (if called outTlv and outTlvSize are also written)
    char ** outTlv,                     // [out] address of serialized TLV
    uint32_t * outTlvSize,              // [out] size of serialized TLV
    const char* lda,                    // [in] [required not null] layer descriptor array
    uint32_t ldaSize,                   // [in] layer descriptor array size
    const char* ro,                     // [opt] read only model fragment
    uint32_t roSize,                    // read only model part size
    const char* state,                  // [opt] state model fragment
    uint32_t stateSize,                 // state model fragment size
    uint32_t scratchSize,               // scratch model fragment size
    uint32_t externalInputSize,         // external input buffer size of model
    uint32_t externalOutputSize,        // external output buffer size of model
    const char* userSignature,          // [opt] optional user content
    uint32_t userSignatureSize          // user content size
)
{
    char versionBuffer[32] = {};
    const uint32_t const256MB = 1 << 28;
    GNA_TLV_EXPECT_NOT_NULL(lda);
    GNA_TLV_EXPECT_NOT_NULL(userAllocatorIn);
    GNA_TLV_EXPECT_NOT_NULL(outTlv);
    GNA_TLV_EXPECT_NOT_NULL(outTlvSize);

    if (stateSize > 0)
    {
        GNA_TLV_EXPECT_NOT_NULL(state);
    }
    if (roSize > 0)
    {
        GNA_TLV_EXPECT_NOT_NULL(ro);
    }
    if (userSignatureSize > 0)
    {
        GNA_TLV_EXPECT_NOT_NULL(userSignature);
    }

    if (ldaSize > const256MB ||
        roSize > const256MB||
        stateSize > const256MB||
        scratchSize > const256MB ||
        externalInputSize > const256MB ||
        userSignatureSize > const256MB)
    {
        return Gna2TlvStatusLengthTooBig;
    }

    const uint32_t numberOfRecordsBeforeLda = 4;
    uint32_t outPadSizeLda = 0;
    const uint32_t sizeOfRecordsBeforeLda = numberOfRecordsBeforeLda * (TLV_EMPTY_RECORD_SIZE + sizeof(uint32_t));
    Gna2TlvStatus status = Gna2TlvImplComputePadSize(sizeOfRecordsBeforeLda, TLV_ANNA_REQUIRED_ALIGNEMENT, TLV_EMPTY_RECORD_SIZE, &outPadSizeLda);
    if (status != Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    const uint32_t sizeOfLdaRecord = ldaSize + roSize + TLV_EMPTY_RECORD_SIZE;
    const uint32_t sizeOfRecordsBeforeState = sizeOfLdaRecord + outPadSizeLda + sizeOfRecordsBeforeLda;
    uint32_t outPadSizeState = 0;
    status = Gna2TlvImplComputePadSize(sizeOfRecordsBeforeState, TLV_ANNA_REQUIRED_ALIGNEMENT, TLV_EMPTY_RECORD_SIZE, &outPadSizeState);
    if (status != Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusOutOfBuffer;
    }

    const uint32_t totalSizeRequired = sizeOfRecordsBeforeState + outPadSizeState + stateSize + userSignatureSize +
        3 * (TLV_EMPTY_RECORD_SIZE) + sizeof(versionBuffer);

    *outTlv = (char*)userAllocatorIn(totalSizeRequired);
    *outTlvSize = totalSizeRequired;

    if(*outTlv == NULL)
    {
        return Gna2TlvUserAllocatorError;
    }
    char* curOutBuffer = *outTlv;

    Gna2TlvImplWrite4BSize(curOutBuffer, TlvTypeScratchSize, scratchSize);
    Gna2TlvImplWrite4BSize(curOutBuffer, TlvTypeLayerDescriptorArraySize, ldaSize);
    Gna2TlvImplWrite4BSize(curOutBuffer, TlvTypeExternalInputBufferSize, externalInputSize);
    Gna2TlvImplWrite4BSize(curOutBuffer, TlvTypeExternalOutputBufferSize, externalOutputSize);

    if(outPadSizeLda != 0)
    {
        Gna2TlvStatus status = Gna2TlvImplPad(curOutBuffer, outPadSizeLda - TLV_EMPTY_RECORD_SIZE);
        if(status != Gna2TlvStatusSuccess)
        {
            return Gna2TlvStatusOutOfBuffer;
        }
        curOutBuffer += outPadSizeLda;
    }

    Gna2TlvImplWriteType(curOutBuffer, TlvTypeLayerDescriptorAndRoArrayData);
    Gna2TlvImplWriteLength(curOutBuffer, ldaSize + roSize);
    Gna2TlvImplCopy(curOutBuffer, lda, ldaSize);
    Gna2TlvImplCopy(curOutBuffer, ro, roSize);

    if (outPadSizeState != 0)
    {
        Gna2TlvStatus status = Gna2TlvImplPad(curOutBuffer, outPadSizeState - TLV_EMPTY_RECORD_SIZE);
        if (status != Gna2TlvStatusSuccess)
        {
            return Gna2TlvStatusOutOfBuffer;
        }
        curOutBuffer += outPadSizeState;
    }

    Gna2TlvImplWriteType(curOutBuffer, TlvTypeStateData);
    Gna2TlvImplWriteLength(curOutBuffer, stateSize);
    Gna2TlvImplCopy(curOutBuffer, state, stateSize);

    Gna2TlvImplWriteType(curOutBuffer, TlvTypeUserSignatureData);
    Gna2TlvImplWriteLength(curOutBuffer, userSignatureSize);
    Gna2TlvImplCopy(curOutBuffer, userSignature, userSignatureSize);

    Gna2GetLibraryVersion(versionBuffer, sizeof(versionBuffer));

    Gna2TlvImplWriteType(curOutBuffer, TlvTypeUserSignatureData);
    Gna2TlvImplWriteLength(curOutBuffer, sizeof(versionBuffer));
    Gna2TlvImplCopy(curOutBuffer, versionBuffer, sizeof(versionBuffer));

    if(curOutBuffer - *outTlv != totalSizeRequired)
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
