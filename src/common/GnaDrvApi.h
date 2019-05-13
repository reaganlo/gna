/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#pragma once

#include "gna-api-status.h"
#include "profiler.h"

#ifdef DRIVER
#   include <ntddk.h>
#else
#   include <initguid.h>
#   include <Windows.h>
#endif // !DRIVER

#ifndef STATUS_T_ALIAS
#define STATUS_T_ALIAS
typedef intel_gna_status_t  status_t;
#endif

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size

typedef UINT8           __1B_RES;   // 1 B of reserved memory

/**
 * Size of GNA (GMM/xNN) configuration data in bytes
 */
#define CFG_SIZE    256

 /**
  * GNA Device Page Table directory
  */
#define     PT_DIR_SIZE             64

  /**
   * Default time in seconds after which driver will try to auto recover
   *  from hardware hang
   */
#define     DRV_RECOVERY_TIMEOUT    60

#define     PT_ENTRY_SIZE           4
   /**
    * Page table entries number
    * There are up to 1024 32-bit pointers in one page in Page Table (L1)
    */
#define     PT_ENTRY_NO             (0x1000 / PT_ENTRY_SIZE)

/* GNASTS register flags */
#define STS_SATURATION_FLAG      0x20000 // WARNING: score has reached the saturation, MUST CLEAR
#define STS_OUTBUFFULL_FLAG      0x10000 // WARNING: hw output buffer is currently full, MUST CLEAR
#define STS_PARAM_OOR_FLAG       0x100   // ERROR: hw parameter out of range
#define STS_VA_OOR_FLAG          0x80    // ERROR: VA out of range
#define STS_UNEXPCOMPL_FLAG      0x40    // ERROR: PCIe error: unexpected completion
#define STS_DMAREQERR_FLAG       0x20    // ERROR: PCIe error: DMA req
#define STS_MMUREQERR_FLAG       0x10    // ERROR: PCIe error: MMU req
#define STS_STATVALID_FLAG       0x08    // compute statistics valid
#define STS_SDTPASUE_FLAG        0x04    // suspended due to pause
#define STS_BPPASUE_FLAG         0x02    // suspended breakpoint match
#define STS_COMPLETED_FLAG       0x01    // scoring completed flag

/* GNA device/driver parameters */
#define GNA_PARAM_DEVICE_ID         (1LLU << 0)
#define GNA_PARAM_RECOVERY_TIMEOUT  (1LLU << 1)
#define GNA_PARAM_IBUFFS            (1LLU << 2)
#define GNA_PARAM_CE_NUM            (1LLU << 3)
#define GNA_PARAM_PLE_NUM           (1LLU << 4)
#define GNA_PARAM_AFE_NUM           (1LLU << 5)
#define GNA_PARAM_HAS_MMU           (1LLU << 6)
#define GNA_PARAM_HWVER             (1LLU << 7)

// disable zero-sized array in struct/union warning
#pragma warning(disable:4200)

// disables anonymous struct/unions warning, useful to flatten structs
#pragma warning(disable:4201)

/******************************************************************************
 *
 * Driver IOCTL's input-output data structures
 *
 ******************************************************************************
 * NOTE: all below IOCTL in/out data type structures have to be 8 B padded
 *          as this is required for x86-x64 spaces cooperation
 *****************************************************************************/

/**
 *  Enumeration of device flavors
 *  Hides gna_device_generation
 */
typedef enum _GnaDeviceType
{
    GNA_HW_NO_DEVICE   = 0x0000,   // No supported device available
    GNA_HW_SKL     = 0x1911,   // GMM Device Sky Lake
    GNA_HW_KBL     = 0x1911,   // GMM Device Kaby Lake // TODO:3: check KBL HW ID
    GNA_HW_CNL     = 0x5A11,   // GNA 0.9 Device Cannon Lake, no CNN support
    GNA_HW_GLK     = 0x3190,   // GNA 1.0 Device Gemini Lake, full featured GNA 1.0
    GNA_HW_EHL     = 0x4511,   // GNA 1.0 Device Elkhartlake, same function set as GLK
    GNA_HW_ICL     = 0x8A11,   // GNA 1.0 Device Ice Lake, same function set as GLK
    GNA_HW_TGL     = 0x9A11,   // GNA 2.0 Device Tiger Lake, full featured GNA 2.0
    GNA_HW_ADL     = 0x46AD,   // GNA 3.0 Device Alder Lake, full featured GNA 3.0
    GNA_HW_SUE_CREEK   = 0xFFFF1,  // GNA 1.0 Embedded Sue Creek
    GNA_HW_JELLYFISH   = 0xFFFF2,  // GNA 2.1 Embedded Jelly Fish
    GNA_HW_ACE_EMBEDDED= 0xFFFF3,  // GNA 3.0 Embedded on Alder Lake PCH/ACE
    GNA_HW_ACE_ANNA    = 0xFFFF4,  // GNA 3.1 ANNA Autonomous Embedded on Alder Lake PCH/ACE

} GnaDeviceType;

/**
 * GNA device capabilities structure
 */
typedef struct _GNA_CPBLTS
{
    UINT32 bldValue;
    UINT32 recoveryTimeout;
    GnaDeviceType deviceType;
} GNA_CPBLTS;

static_assert(12 == sizeof(GNA_CPBLTS), "Invalid size of GNA_CPBLTS");

/**
 * Calculate Control flags
 */
typedef struct _CTRL_FLAGS
{
    UINT32      activeListOn : 1;    // active list mode (0:disabled, 1:enabled)
    UINT32      gnaMode : 2;         // GNA operation mode (0:GMM, 1:xNN)
    UINT32      ddiVersion : 21;
    UINT32      hwPerfEncoding : 8;
    union
    {
        UINT32      xnnLyrDscSize;   // backward compatibility: size of layer descriptors sent
        UINT32      layerCount;
    };
} CTRL_FLAGS;                       // Control flag

static_assert(8 == sizeof(CTRL_FLAGS), "Invalid size of CTRL_FLAGS");

/**
 * CALCULATE request data with output information.
 * NOTE: always include performance results
 * this allow to use PROFILED library with NON-PROFILED driver and vice versa
 */
typedef struct _GNA_CALC_IN
{
    /* input part */
    CTRL_FLAGS          ctrlFlags;        // scoring mode

    union {
        UINT8           config[CFG_SIZE]; // configuration data for GMM or xNN
        struct {
            UINT32      configBase;       // layer base / offset to gmm descriptor
            UINT64      bufferCount;      // number of buffers lying outside this structure
        };
    };

    /* output part*/
    perf_drv_t          drvPerf;          // driver level performance profiling results
    perf_hw_t           hwPerf;           // hardware level performance results
    UINT32              status;           // status_t (LEGACY) or GNASTS value after scoring

    UINT32              pad;            // 4 B padding to multiple 8 B size

    UINT8               buffers[];        // memory buffers with patches
} GNA_CALC_IN, *PGNA_CALC_IN;       // CALCULATE IOCTL - Input/output data

static_assert(312 == sizeof(GNA_CALC_IN), "Invalid size of GNA_CALC_IN");

/*
 * User buffer identified by memory id
 * List of such buffers are received in WRITE request to driver
 * Each buffer is added to MMU according to it's offset and size
 * Each buffer may contain patches that driver will apply to the memory before starting GNA
 */
typedef struct _GNA_MEMORY_BUFFER
{
    UINT64 memoryId;
    UINT64 offset;
    UINT64 size;
    UINT64 patchCount;
} GNA_MEMORY_BUFFER, *PGNA_MEMORY_BUFFER;

static_assert(32 == sizeof(GNA_MEMORY_BUFFER), "Invalid size of GNA_MEMORY_BUFFER");

/* Patch structure describes memory location that has to be patched before request
 * Memory is patched according to provided data and it's size
 * List of such patches are received in WRITE request to driver
 * Each patch is linked to memory described by GNA_MEMORY_BUFFER
 */
typedef struct _GNA_MEMORY_PATCH
{
    UINT64 offset;
    UINT64 size;
    UINT8 data[];
} GNA_MEMORY_PATCH, *PGNA_MEMORY_PATCH;

static_assert(16 == sizeof(GNA_MEMORY_PATCH), "Invalid size of GNA_MEMORY_PATCH");

/**
 * Minimum Size of GNA (GMM/xNN) request in bytes
 */
#define REQUEST_SIZE                sizeof(GNA_CALC_IN)

 /**
  * Size of xNN layer descriptor in bytes
  */
#define XNN_LYR_DSC_SIZE            (128)

  /**
   * Size of GMM config in bytes
   */
#define GMM_CFG_SIZE                (128)

#pragma pack ()

