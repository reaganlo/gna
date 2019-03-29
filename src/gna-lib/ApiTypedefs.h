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

/**
 Aliases for official API types
 */

#pragma once

#include "gna2-api.h"

#ifndef STATUS_T_ALIAS
#define STATUS_T_ALIAS
typedef enum GnaStatus ApiStatus;
#endif

typedef enum GnaBiasMode ApiBiasMode;
typedef enum GnaDataType ApiDataType;
typedef enum GnaDeviceGeneration DeviceGeneration;
typedef enum GnaDeviceVersion DeviceVersion;
typedef enum GnaErrorType ModelErrorType;
typedef enum GnaInstrumentationMode HwInstrumentationMode;
typedef enum GnaInstrumentationUnit InstrumentationUnit;
typedef enum GnaOperationType ApiOperationType;
typedef enum GnaPoolingMode ApiPoolingMode;
typedef enum GnaTensorMode ApiTensorMode;
typedef struct GnaDrvPerfResults DrvPerfResults;
typedef struct GnaModel ApiModel;
typedef struct GnaModelEmbeddedHeader ModelEmbeddedHeader;
typedef struct GnaOperation ApiOperation;
typedef struct GnaPerfResults PerfResults;
typedef struct GnaShape ApiShape;
typedef struct GnaTensor ApiTensor;

/**
 Verifies data sizes used in the API and GNA hardware

 @note If data sizes in an application using API differ from data sizes
       in the API library implementation, scoring will not work properly.
 */
static_assert(1 == sizeof(int8_t), "Invalid size of int8_t");
static_assert(2 == sizeof(int16_t), "Invalid size of int16_t");
static_assert(4 == sizeof(int32_t), "Invalid size of int32_t");
static_assert(1 == sizeof(uint8_t), "Invalid size of uint8_t");
static_assert(2 == sizeof(uint16_t), "Invalid size of uint16_t");
static_assert(4 == sizeof(uint32_t), "Invalid size of uint32_t");
