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

#include <stddef.h>
#include <stdint.h>

GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvCheckValid(const Gna2TlvRecord * tlvRecord, const char* tlvBlobEnd)
{
    GNA2_TLV_EXPECT_NOT_NULL(tlvRecord);

    if (tlvBlobEnd < tlvRecord->value)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    if((size_t)(tlvBlobEnd - tlvRecord->value) < tlvRecord->length)
    {
        return Gna2TlvStatusOutOfBuffer;
    }
    return Gna2TlvStatusSuccess;
}

GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvFindInArray(
    const char* tlvArrayBegin,          // Address of the first TLV record e.g. TLV formatted GNA model address
    uint32_t tlvArraySize,              // Byte size of TLV records in memory e.g., size of the TLV formatted GNA model
    const Gna2TlvType tlvTypeToFind,    // TLV type of the record to find
    uint32_t *outValueLength,           // TLV length of the record found
    void **outValue                     // TLV value address of the record found
)
{
    GNA2_TLV_EXPECT_NOT_NULL(tlvArrayBegin);
    GNA2_TLV_EXPECT_NOT_NULL(outValueLength);
    GNA2_TLV_EXPECT_NOT_NULL(outValue);

    const char* const tlvArrayEnd = tlvArrayBegin + tlvArraySize;
    while (tlvArrayBegin < tlvArrayEnd)
    {
        const Gna2TlvRecord* const currentRecord = (const Gna2TlvRecord*)tlvArrayBegin;
        if (Gna2TlvStatusSuccess != Gna2TlvCheckValid(currentRecord, tlvArrayEnd))
        {
            return Gna2TlvStatusTlvReadError;
        }
        if (tlvTypeToFind == currentRecord->type)
        {
            *outValue = (void *)(currentRecord->value);
            *outValueLength = currentRecord->length;
            return Gna2TlvStatusSuccess;
        }
        tlvArrayBegin = currentRecord->value + currentRecord->length;
    }
    *outValue = NULL;
    *outValueLength = 0;
    return Gna2TlvStatusNotFound;
}

GNA2_TLV_LINKAGE Gna2TlvStatus Gna2TlvVerifyVersionAndCohesion(
    const char* tlvArrayBegin,          // Address of the first TLV record e.g. TLV formatted GNA model address
    uint32_t tlvArraySize               // Byte size of TLV records in memory e.g., size of the TLV formatted GNA model
)
{
    void * version = NULL;
    uint32_t length = 0;
    Gna2TlvStatus status = Gna2TlvFindInArray(tlvArrayBegin, tlvArraySize, Gna2TlvTypeTlvVersion, &length, &version);
    if (status != Gna2TlvStatusSuccess)
    {
        return status == Gna2TlvStatusNotFound ? Gna2TlvStatusVersionNotFound : status;
    }
    if(length != GNA2_TLV_VERSION_VALUE_LENGTH || (*(uint32_t*)version) != GNA2_TLV_VERSION)
    {
        return Gna2TlvStatusVersionNotSupported;
    }
    tlvArrayBegin += GNA2_TLV_VERSION_RECORD_SIZE;
    tlvArraySize -= GNA2_TLV_VERSION_RECORD_SIZE;
    status = Gna2TlvFindInArray(tlvArrayBegin, tlvArraySize, Gna2TlvTypeTlvVersion, &length, &version);
    if(status == Gna2TlvStatusNotFound)
    {
        return Gna2TlvStatusSuccess;
    }
    if(status == Gna2TlvStatusSuccess)
    {
        return Gna2TlvStatusSecondaryVersionFound;
    }
    return status;
}
#endif // __GNA2_TLV_ANNA_READER_H

/**
 @}
 @}
 */
