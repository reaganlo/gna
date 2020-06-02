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
 @file gna2-tlv-anna-reader.h
 @brief Gaussian and Neural Accelerator (GNA) 3.1 TLV Anna reader.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA3_API Gaussian and Neural Accelerator (GNA) 3.1 API.
 @{

 ******************************************************************************

 @addtogroup GNA2_MODEL_EXPORT_API Embedded Model Export for Anna.

 @{
 *****************************************************************************/

#ifndef __GNA2_TLV_ANNA_READER_H
#define __GNA2_TLV_ANNA_READER_H

#include "gna2-tlv-anna.h"

#include <stdint.h>


Gna2TlvStatus Gna2TlvImplGetLength(const char * tlvRecord, const uint32_t tlvBlobSize, uint32_t * outLength)
{
    if (tlvRecord == NULL || outLength == NULL)
    {
        Gna2TlvStatusNullNotAllowed;
    }
    if (tlvBlobSize < TLV_EMPTY_RECORD_SIZE)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    const uint32_t length = *(Gna2TlvLength*)(tlvRecord + sizeof(Gna2TlvType));
    if(tlvBlobSize - TLV_EMPTY_RECORD_SIZE < *outLength)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    *outLength = length;
    return Gna2TlvStatusSuccess;
}

Gna2TlvStatus Gna2TlvImplGetNextOffset(const char * tlvRecord,
    uint32_t tlvBlobSize,
    uint32_t* tlvNextRecordOffset)
{
    if (tlvNextRecordOffset == NULL)
    {
        return Gna2TlvStatusNullNotAllowed;
    }
    uint32_t currentRecordLength = 0;
    const Gna2TlvStatus status = Gna2TlvImplGetLength(tlvRecord, tlvBlobSize, &currentRecordLength);

    if (Gna2TlvStatusSuccess != status)
    {
        return status;
    }

    *tlvNextRecordOffset = currentRecordLength + TLV_EMPTY_RECORD_SIZE;
    return Gna2TlvStatusSuccess;
}

inline Gna2TlvStatus Gna2TlvFindInArray(
    const char* tlvArrayBegin,          // Address of the first TLV record e.g. TLV formated GNA model address
    uint32_t tlvArraySize,              // Byte size of TLV records in memory e.g., size of the TLV formatted GNA model
    const Gna2TlvType tlvTypeToFind,    // TLV type of the record to find
    uint32_t *outValueLength,           // TLV length of the record found
    void **outValue                     // TLV value address of the record found
)
{
    if(tlvArrayBegin == NULL || outValueLength == NULL || outValue == NULL)
    {
        return Gna2TlvStatusNullNotAllowed;
    }

    while (tlvArraySize >= TLV_EMPTY_RECORD_SIZE)
    {
        if (tlvTypeToFind == *((Gna2TlvTypeC*)tlvArrayBegin))
        {
            uint32_t length = 0;
            if (Gna2TlvStatusSuccess != Gna2TlvImplGetLength(tlvArrayBegin, tlvArraySize, &length))
            {
                return Gna2TlvStatusOutOfBuffer;
            }
            *outValue = (void *)(tlvArrayBegin + TLV_EMPTY_RECORD_SIZE);
            *outValueLength = length;
            return Gna2TlvStatusSuccess;
        }
        uint32_t nextRecordOffset = 0;
        if(Gna2TlvStatusSuccess != Gna2TlvImplGetNextOffset(tlvArrayBegin, tlvArraySize, &nextRecordOffset))
        {
            break;
        }
        tlvArraySize -= nextRecordOffset;
        tlvArrayBegin += nextRecordOffset;
    }
    *outValue = NULL;
    *outValueLength = 0;
    return Gna2TlvStatusNotFound;
}


#endif // __GNA2_TLV_ANNA_READER_H

/**
 @}
 @}
 */
