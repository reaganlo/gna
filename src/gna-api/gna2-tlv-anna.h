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

#include <stdbool.h>

typedef uint32_t Gna2TlvType;
typedef const Gna2TlvType Gna2TlvTypeC;
typedef uint32_t Gna2TlvLength;

#define TLV_IMPL_CHAR_TO_TYPE(NAME) (*((Gna2TlvTypeC*)NAME))

static Gna2TlvTypeC TlvTypeLayerDescriptorArraySize = TLV_IMPL_CHAR_TO_TYPE("LDAS");
static Gna2TlvTypeC TlvTypeLayerDescriptorAndRoArrayData = TLV_IMPL_CHAR_TO_TYPE("L&RD");
static Gna2TlvTypeC TlvTypeStateData = TLV_IMPL_CHAR_TO_TYPE("STTD");
static Gna2TlvTypeC TlvTypeScratchSize = TLV_IMPL_CHAR_TO_TYPE("SCRS");
static Gna2TlvTypeC TlvTypePadding = TLV_IMPL_CHAR_TO_TYPE("PAD\0");
static Gna2TlvTypeC TlvTypeExternalInputBufferSize = TLV_IMPL_CHAR_TO_TYPE("ExIS");
static Gna2TlvTypeC TlvTypeExternalOutputBufferSize = TLV_IMPL_CHAR_TO_TYPE("ExOS");

static Gna2TlvTypeC TlvTypeUserSignatureData = TLV_IMPL_CHAR_TO_TYPE("USRD");

#define TLV_ANNA_REQUIRED_ALIGNEMENT 64
#define TLV_EMPTY_RECORD_SIZE (sizeof(Gna2TlvLength) + sizeof(Gna2TlvType))


typedef int32_t Gna2TlvStatus;

static const int32_t Gna2TlvStatusSuccess = 0;
static const int32_t Gna2TlvStatusUnknownError = 0x1;
static const int32_t Gna2TlvStatusNotFound = 0x10;
static const int32_t Gna2TlvStatusNullNotAllowed = 0x20;
static const int32_t Gna2TlvStatusOutOfBuffer = 0x40;
static const int32_t Gna2TlvStatusNotSupported = 0x80;
static const int32_t Gna2TlvStatusLengthTooBig = 0x100;
static const int32_t Gna2TlvUserAllocatorError = 0x200;

typedef void* Gna2TlvAllocator(uint32_t);

#endif // __GNA2_TLV_ANNA_H

/**
 @}
 @}
 */
