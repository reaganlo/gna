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

#ifdef DRIVER
#   include <ntddk.h>
#else
#   include <initguid.h>
#   include <Windows.h>
#endif // !DRIVER

#include "gna-api-status.h"
#include "profiler.h"

#if defined(_DEBUG) || !defined(DRIVER)
#define DRV_DEBUG_INTERFACE
#endif

#ifndef STATUS_T_ALIAS
#define STATUS_T_ALIAS
typedef intel_gna_status_t  status_t;
#endif

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size


typedef UINT8           __1B_RES;   // 1 B of reserved memory

/**
 * GNA max models limit for single application
 */
#define     APP_MODELS_LIMIT         32

/**
 * GNA Device Page Table directory
 */
#define     PT_DIR_SIZE             64


/**
 * Time in seconds after which driver will try to auto recover
 *  from hardware hang
 */
#define     DRV_RECOVERY_TIMEOUT    60
// TODO: add registry value to enable manipulation of recovery time

/**
 * Page table entries number
 * There are up to 1024 32-bit pointers in one page in Page Table (L1)
 */
#define     PT_ENTRY_NO             (0x1000 / 4)

/**
 * Whole Page table size (L2)
 */
#define     PT_SIZE                 ((PT_DIR_SIZE + 1) * PT_ENTRY_NO)


#pragma warning(disable:4201)       // disables anonymous struct/unions warning, useful to flatten structs

/**
 * Define an Interface Guid so that app can find the device and talk to it.
 */

// {8113B324-9F9B-4B9F-BF55-1342A58593DC}
DEFINE_GUID(GUID_DEVINTERFACE_GNA_DRV,
    0x8113b324, 0x9f9b, 0x4b9f, 0xbf, 0x55, 0x13, 0x42, 0xa5, 0x85, 0x93, 0xdc);

// {608D09B8-41BC-4079-A040-1EE3F48483DD}
DEFINE_GUID(GUID_DEVINTERFACE_GMM_DRV,
    0x608d09b8, 0x41bc, 0x4079, 0xa0, 0x40, 0x1e, 0xe3, 0xf4, 0x84, 0x83, 0xdd);


/******************************************************************************
 *
 * Driver IOCTL's input-output data structures
 *
 ******************************************************************************
 * NOTE: all below IOCTL in/out data type structures have to be 8 B padded
 *          as this is required for x86-x64 spaces cooperation
 *****************************************************************************/

/**
 * MEM_MAP IOCTL - input data.
 * Size:    8 B
 */
typedef struct _GNA_MM_IN
{
    UINT64 model_id;     // Length of the application buffer

} GNA_MM_IN, *PGNA_MM_IN;           // MEM_MAP IOCTL - input data

static_assert(8 == sizeof(GNA_MM_IN), "Invalid size of GNA_MM_IN");

/**
 * READ_PGDIR IOCTL - output data.
 * Size:    266 768 B
 */
typedef struct _GNA_PGDIR_OUT
{
    UINT64              ptCount;    // Number of L1 pages allocated by the driver
    UINT64              l1PhysAddr[PT_DIR_SIZE+1];// physical addresses of allocated pages
    UINT32              l2PhysAddr[PT_SIZE];// physical addresses of page entries

} GNA_PGDIR_OUT, *PGNA_PGDIR_OUT;         // READ_PGDIR IOCTL - output data (debug mode only)

static_assert(266768 == sizeof(GNA_PGDIR_OUT), "Invalid size of GNA_PGDIR_OUT");

/**
 *  Enumeration of device flavors
 *  Hides gna_device_kind
 */
typedef enum _GnaDeviceType {
    GNA_SUE_CREEK,
    GNA_SUE_CREEK_2,
    GNA_CANNONLAKE,
    GNA_GEMINILAKE,
    GNA_ICELAKE,
    GNA_LAKEFIELD,
    GNA_TIGERLAKE,

    GNA_NUM_DEVICE_TYPES,
} GnaDeviceType;

/**
 * GNA device capabilities structure
 * Size: 4B
 */
typedef struct _GNA_CPBLTS
{
    UINT32 hwInBuffSize;
    GnaDeviceType device_type;
} GNA_CPBLTS;

static_assert(8 == sizeof(GNA_CPBLTS), "Invalid size of GNA_CPBLTS");

/**
 * Calculate Control flags
 * Size:    8 B
 */
typedef struct _CTRL_FLAGS
{
    UINT32      activeListOn    :1; // 00:00 - active list mode (0:disabled, 1:enabled)
    UINT32      gnaMode         :2; // 01:02 - GNA operation mode (0:GMM, 1:xNN)
    UINT32      layerIndex      :14;
    UINT32      layerCount      :14;
    UINT32      _rsvd           :1;
} CTRL_FLAGS;                       // Control flag

static_assert(4 == sizeof(CTRL_FLAGS), "Invalid size of CTRL_FLAGS");

/**
 * Structure used to send to driver which buffer addresses to overwrite according to
 * score request configuration
 */
typedef struct _GNA_BUFFER_DESCR
{
    UINT32 offset; // points where to write the value (refers to user memory base address)
    UINT32 value; // address offset of the buffer (refers to user memory base address)
} GNA_BUFFER_DESCR, *PGNA_BUFFER_DESCR;

static_assert(8 == sizeof(GNA_BUFFER_DESCR), "Invalid size of GNA_BUFFER_DESCR");

typedef struct _NNOP_TYPE_DESCR
{
    UINT32 offset;
    UINT8  value;
} NNOP_TYPE_DESCR, *PNNOP_TYPE_DESCR;

static_assert(5 == sizeof(NNOP_TYPE_DESCR), "Invalid size of NNOP_TYPE_DESCR");

/**
 * _offset member points where to write the _value (refers to user memory base address)
 */
typedef struct _XNN_ACTIVE_LIST_DESCR
{
    UINT32 act_list_buffer_offset;
    UINT32 act_list_buffer_value;
    UINT32 act_list_n_elems_offset;
    UINT16 act_list_n_elems_value;
} XNN_ACTIVE_LIST_DESCR, *PXNN_ACTIVE_LIST_DESCR;

static_assert(14 == sizeof(XNN_ACTIVE_LIST_DESCR), "Invalid size of XNN_ACTIVE_LIST_DESCR");

/**
* _offset member points where to write the _value (refers to user memory base address)
*/
typedef struct _GMM_ACTIVE_LIST_DESCR
{
    UINT32 asladdr_offset;
    UINT32 asladdr_value;
    UINT32 astlistlen_offset;
    UINT32 astlistlen_value;
    UINT32 gmmscrlen_offset;
    UINT32 gmmscrlen_value;
} GMM_ACTIVE_LIST_DESCR, *PGMM_ACTIVE_LIST_DESCR;

static_assert(24 == sizeof(GMM_ACTIVE_LIST_DESCR), "Invalid size of GMM_ACTIVE_LIST_DESCR");

typedef struct _REQ_CONFIG_DESCR
{
    UINT32  requestConfigId;
    UINT32  buffersCount;
    UINT32  nnopTypesCount;
    UINT32  xnnActiveListsCount;
    UINT32  gmmActiveListsCount;
} REQ_CONFIG_DESCR, *PREQ_CONFIG_DESCR;

static_assert(20 == sizeof(REQ_CONFIG_DESCR), "Invalid size of REQ_CONFIG_DESCR");

/**
 * CALCULATE request data with output information.
 * Size:    52 B
 * NOTE: always include performance results
 * this allow to use PROFILED library with NON-PROFILED driver and vice versa
 */
typedef struct _GNA_CALC_IN
{
    UINT64              modelId;        // model identifier
    CTRL_FLAGS          ctrlFlags;      // scoring mode
    perf_drv_t          drvPerf;        // driver level performance profiling results
    perf_hw_t           hwPerf;         // hardware level performance results
    UINT8               hwPerfEncoding; // hardware level performance encoding type
    status_t            status;         // status of scoring
    REQ_CONFIG_DESCR    reqCfgDescr;
} GNA_CALC_IN, *PGNA_CALC_IN;       // CALCULATE IOCTL - Input data

static_assert(77 == sizeof(GNA_CALC_IN), "Invalid size of GNA_CALC_IN");

/**
 * Minimum Size of GNA (GMM/xNN) request in bytes
 */
#define REQUEST_SIZE                sizeof(GNA_CALC_IN)

/**
 * Minimum Size of xNN layer descriptors in bytes (at least 1 layer)
 */
#define XNN_LYR_DSC_SIZE            (128)

/**
 * READ_REG IOCTL - input data
 * Size:    8 B
 */
typedef struct _PGNA_READREG_IN
{
    UINT32              mbarIndex;  // Index of MBAR
    UINT32              regOffset;  // Register offset

} GNA_READREG_IN, *PGNA_READREG_IN;// READ_REG IOCTL - input data

static_assert(8 == sizeof(GNA_READREG_IN), "Invalid size of GNA_READREG_IN");

/**
 * READ_REG IOCTL - output data
 * Size:    8 B
 */
typedef struct _GNA_READREG_OUT
{
    UINT32              regValue;   // Register value
    UINT32              __res;      // 4 B padding to multiple 8 B size

} GNA_READREG_OUT, *PGNA_READREG_OUT;//READ_REG IOCTL - output data

static_assert(8 == sizeof(GNA_READREG_OUT), "Invalid size of GNA_READREG_OUT");

/**
 * WRITE_REG ioctl - input data
 * Size:    16 B
 */
typedef struct _GNA_WRITEREG_IN
{
    UINT32              mbarIndex;  // Index of MBAR
    UINT32              regOffset;  // Register offset
    UINT32              regValue;   // Register value
    UINT32              __res;      // 4 B padding to multiple 8 B size

} GNA_WRITEREG_IN, *PGNA_WRITEREG_IN;// WRITE_REG ioctl - input data

static_assert(16 == sizeof(GNA_WRITEREG_IN), "Invalid size of GNA_WRITEREG_IN");

/******************************************************************************
 *
 * Driver IOCTL interface
 *
 *****************************************************************************/
#define FILE_DEVICE_PCI_GNA 0x8000
#define GNA_IOCTL_MEM_MAP   CTL_CODE(FILE_DEVICE_PCI_GNA, 0x900, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define GNA_IOCTL_MEM_UNMAP CTL_CODE(FILE_DEVICE_PCI_GNA, 0x901, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_WAKEUP_HW CTL_CODE(FILE_DEVICE_PCI_GNA, 0x905, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_CPBLTS    CTL_CODE(FILE_DEVICE_PCI_GNA, 0x902, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_NOTIFY    CTL_CODE(FILE_DEVICE_PCI_GNA, 0x907, METHOD_NEITHER, FILE_ANY_ACCESS)

#ifdef  DRV_DEBUG_INTERFACE

#define GNA_IOCTL_READ_REG   CTL_CODE(FILE_DEVICE_PCI_GNA, 0x903, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_WRITE_REG  CTL_CODE(FILE_DEVICE_PCI_GNA, 0x904, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_READ_PGDIR CTL_CODE(FILE_DEVICE_PCI_GNA, 0x906, METHOD_BUFFERED, FILE_ANY_ACCESS)

#endif // DRV_DEBUG_INTERFACE

#pragma pack ()

