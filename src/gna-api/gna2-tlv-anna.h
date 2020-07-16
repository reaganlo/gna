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
 @file gna2-tlv-anna.h
 @brief Gaussian and Neural Accelerator (GNA) 3.1 Anna Header.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA3_API Gaussian and Neural Accelerator (GNA) 3.1 API.
 @{

 ******************************************************************************

 @addtogroup GNA2_MODEL_EXPORT_API Embedded Model Export for Anna.

 @{
 *****************************************************************************/

#ifndef __GNA2_TLV_ANNA_H
#define __GNA2_TLV_ANNA_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

// TODO: 3: consider removing typedefs
typedef uint32_t Gna2TlvType;
typedef uint32_t Gna2TlvLength;

typedef struct
{
    Gna2TlvType type;
    Gna2TlvLength length;
    char value[];
} Gna2TlvRecord;

#define GNA2_TLV_LINKAGE static

static_assert(sizeof(Gna2TlvRecord) == 8, "Wrong size of Gna2TlvRecord");

#define GNA2_TLV_IMPL_CHAR_TO_TYPE(CSTR) (*((const Gna2TlvType*)CSTR))

#define Gna2TlvTypeLayerDescriptorArraySize GNA2_TLV_IMPL_CHAR_TO_TYPE("LDAS")
#define Gna2TlvTypeLayerDescriptorAndRoArrayData GNA2_TLV_IMPL_CHAR_TO_TYPE("L&RD")
#define Gna2TlvTypeStateData GNA2_TLV_IMPL_CHAR_TO_TYPE("STTD")
#define Gna2TlvTypeScratchSize GNA2_TLV_IMPL_CHAR_TO_TYPE("SCRS")
#define Gna2TlvTypePadding GNA2_TLV_IMPL_CHAR_TO_TYPE("PAD\0")
#define Gna2TlvTypeExternalInputBufferSize GNA2_TLV_IMPL_CHAR_TO_TYPE("ExIS")
#define Gna2TlvTypeExternalOutputBufferSize GNA2_TLV_IMPL_CHAR_TO_TYPE("ExOS")

#define Gna2TlvTypeUserData GNA2_TLV_IMPL_CHAR_TO_TYPE("USRD")
#define Gna2TlvTypeGnaLibraryVersionString GNA2_TLV_IMPL_CHAR_TO_TYPE("GNAV")
#define Gna2TlvTypeTlvVersion GNA2_TLV_IMPL_CHAR_TO_TYPE("TLVV")

#define GNA2_TLV_ANNA_REQUIRED_ALIGNEMENT 64
#define GNA2_TLV_LENGTH_SIZE sizeof(Gna2TlvLength)
#define GNA2_TLV_EMPTY_RECORD_SIZE (GNA2_TLV_LENGTH_SIZE + sizeof(Gna2TlvType))
#define GNA2_TLV_VERSION 1
#define GNA2_TLV_VERSION_VALUE_LENGTH sizeof(uint32_t)
#define GNA2_TLV_VERSION_RECORD_SIZE (GNA2_TLV_EMPTY_RECORD_SIZE + GNA2_TLV_VERSION_VALUE_LENGTH)

// TODO: 3: consider removing typedefs
typedef int32_t Gna2TlvStatus;

#define Gna2TlvStatusSuccess 0
#define Gna2TlvStatusUnknownError 0x1
#define Gna2TlvStatusNotFound 0x10
#define Gna2TlvStatusNullNotAllowed 0x20
#define Gna2TlvStatusOutOfBuffer 0x40
#define Gna2TlvStatusNotSupported 0x80
#define Gna2TlvStatusLengthTooBig 0x100
#define Gna2TlvStatusLengthOver256MB 0x101
#define Gna2TlvStatusUserAllocatorError 0x200
#define Gna2TlvStatusTlvReadError 0x300
#define Gna2TlvStatusSecondaryVersionFound 0x301
#define Gna2TlvStatusVersionNotFound 0x302
#define Gna2TlvStatusVersionNotSupported 0x303

typedef void* Gna2TlvAllocator(uint32_t);

#define GNA2_TLV_EXPECT_NOT_NULL(ADDRESS) {if((ADDRESS) == NULL) return Gna2TlvStatusNullNotAllowed;}
#define GNA2_TLV_EXPECT_NOT_NULL_IF_SIZE_NZ(DATASIZE, ADDRESS) {if((DATASIZE) != 0) GNA2_TLV_EXPECT_NOT_NULL(ADDRESS)}
#define GNA2_TLV_EXPECT_LENGTH_UP_TO_256MB(LENGTH) {if((LENGTH) > (1 << 28)) return Gna2TlvStatusLengthOver256MB;}

#endif // __GNA2_TLV_ANNA_H

/**
 @}
 @}
 */
