/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.
    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/******************************************************************************
 *
 * GNA 3.0 API
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * API Definition
 *
 *****************************************************************************/

#ifndef __GNA_API_H
#define __GNA_API_H

#include <stdint.h>

#if !defined(_WIN32)
#include <assert.h>
#endif

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

/******************************************************************************
 * GNA Capabilities API
 *****************************************************************************/
typedef enum _api_version
{
    GNA_API_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
    GNA_API_1_0,
    GNA_API_2_0,
    //GNA_API_2_1_S, // TODO: use GNA_API_3_0
    GNA_API_3_0,

    GNA_API_VERSION_COUNT
} gna_api_version;

// TODO:3: rename to generation or remove and use driver device types
typedef enum _device_generation
{
    GNA_DEVICE_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
    GMM_DEVICE,
    GNA_0_9,                // GNA 0.9 Device Cannon Lake (CNL), no CNN support
    GNA_1_0,                // GNA 1.0 Device Gemini Lake (GLK), full featured GNA 1.0
                            // GNA 1.0 Device Ice Lake (ICL), same function set as GLK
    GNA_1_0_EMBEDDED,       // GNA 1.0 Embedded Sue Creek (SUE)
    GNA_2_0,                // GNA 2.0 Device Tiger Lake, full featured GNA 2.0 (TGL)
    GNA_2_1_EMBEDDED,       // GNA 2.1 Embedded Jelly Fish (JFL)
    GNA_3_0,                // GNA 3.0 Device Alder Lake, full featured GNA 3.0 (ADL)
    GNA_3_5,                // GNA 3.5 Device Meteor Lake, full featured GNA 3.5 (MTL)
    GNA_3_0_EMBEDDED,       // GNA 3.0 Embedded on Alder Lake PCH/ACE
    GNA_3_1_AUTONOMUS,      // GNA 3.1 ANNA Autonomous Embedded on Alder Lake PCH/ACE
    GNA_DEVICE_COUNT
} gna_device_generation;

/**
 *  Enumeration of device flavors
 */
typedef enum _gna_device_version
{
    GNA_UNSUPPORTED  = (int)GNA_NOT_SUPPORTED, // No supported hardware device available.
    GNA_GMM          = 0x01,            // GMM Device
    GNA_0x9          = 0x09,            // GNA 0.9 Device, no CNN support
    GNA_1x0          = 0x10,            // GNA 1.0 Device, full featured GNA 1.0
    GNA_2x0          = 0x20,            // GNA 2.0 Device, full featured GNA 2.0
    GNA_3x0          = 0x30,            // GNA 3.0 Device, full featured GNA 3.0
    GNA_3x5          = 0x35,            // GNA 3.5 Device, full featured GNA 3.5
    GNA_EMBEDDED_1x0 = 0x10E,           // GNA 1.0 Embedded
    GNA_EMBEDDED_2x1 = 0x20E,           // GNA 2.1 Embedded
    GNA_EMBEDDED_3x0 = 0x30E,           // GNA 3.0 Embedded PCH
    GNA_EMBEDDED_3x1 = 0x31A,           // GNA 3.1 Autonomous Embedded on PCH,
    GNA_SOFTWARE_EMULATION = GNA_DEFAULT, // Software emulation fall-back will be used.

} gna_device_version;


/** Maximum number of requests that can be enqueued before retrieval */
const uint32_t GNA_REQUEST_QUEUE_LENGTH = 64;

/******************************************************************************
 * GNA Utilities API
 *****************************************************************************/

/**
 * Rounds a number up, to the nearest multiple of significance
 * Used for calculating the memory sizes of GNA data buffers
 *
 * @param number        Memory size or a number to round up.
 * @param significance  Informs the function how to round up. The function "ceils"
 *                      the number to the lowest possible value divisible by "significance".
 * @return Rounded integer value.
 * @deprecated          Will be removed in next release.
 */
#define ALIGN(number, significance)   ((((number) + (significance) - 1) / (significance)) * (significance))

/**
 * Rounds a number up, to the nearest multiple of 64
 * Used for calculating memory sizes of GNA data arrays
 * @deprecated          Will be removed in next release.
 */
#define ALIGN64(number)   ALIGN(number, 64)

/**
 * Verifies data sizes used in the API and GNA hardware
 *
 * NOTE: If data sizes in an application using API differ from data sizes
 *       in the API library implementation, scoring will not work properly
 */
static_assert(1 == sizeof(int8_t), "Invalid size of int8_t");
static_assert(2 == sizeof(int16_t), "Invalid size of int16_t");
static_assert(4 == sizeof(int32_t), "Invalid size of int32_t");
static_assert(1 == sizeof(uint8_t), "Invalid size of uint8_t");
static_assert(2 == sizeof(uint16_t), "Invalid size of uint16_t");
static_assert(4 == sizeof(uint32_t), "Invalid size of uint32_t");

#endif  // ifndef __GNA_API_H
