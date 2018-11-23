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

/*++

Module Name:

    driver.h

Abstract:

    This file contains the driver common types.

Environment:

    Kernel-mode Driver Framework

--*/

#if !defined (_DRIVER_H)
#define _DRIVER_H

#define INITGUID

#include <ntddk.h>
#include <wdf.h>

#if 10 == TARGET_WIN_VER
#	if NTDDI_VERSION < 0x0A000000
#	error Target Windows 10 version is not set properly
#	endif
#elif 63 == TARGET_WIN_VER
#	if NTDDI_VERSION != 0x06030000
#	error Target Windows 8.1 version is not set properly
#	endif
#else
#	error Target Windows version is not set
#endif

#include "Trace.h"
#include "HwRegisters.h"
#include "HwDescriptors.h"

// alias for API success status
#define     GNA_SUCCESS GNA_SUCCESS

/******************************************************************************
 * Application specific
 *****************************************************************************/

/**
 * Page Table Directory
 */
typedef struct _PT_DIR
{
    WDFCOMMONBUFFER     commBuff;
    PUCHAR              commBuffVa;
    PHYSICAL_ADDRESS    commBuffLa;

} PT_DIR, *P_PT_DIR;                // Page Table Directory

/**
 * HW Descriptor context
 */
typedef struct _HW_DESC
{
    WDFCOMMONBUFFER     buffer;     // descriptor common buffer memory
    PDESCRIPTOR         va;         // descriptor memory Virtual address (for driver)
    PHYSICAL_ADDRESS    la;         // descriptor memory physical address (for device)

} HW_DESC, *PHW_DESC;               //  HW Descriptor context

typedef struct _MEMORY_CTX
{
    LIST_ENTRY    listEntry;
    UINT64        memoryId;                 // Back-reference memory id
    PMDL          pMdl;                     // Pointer to MDL object used with MmLock/MmUnlock
    WDFREQUEST    mmapRequest;              // Memory map request to be completed on memory release
    PT_DIR        ptDir[PT_DIR_SIZE + 1];   // page table directory
    ULONG         pageTableCount;           // Number of actually used entries in page Tables.
    HW_DESC       desc;                     // hardware descriptor context
    PVOID         userMemoryBaseVA;         // User memory virtual address
    UINT32        userMemorySize;           // Size of user memory

} MEMORY_CTX, *PMEMORY_CTX;               // Memory context

/**
 * API 1 client context - basically it's a reduced memory context
 */
typedef struct _APP_CTX
{
    PMDL                pMdl;       // Pointer to MDL object used with MmLock/MmUnlock
    BOOLEAN             memLocked;  // Indicates whether the memory is successfully locked.
    PT_DIR              ptDir[PT_DIR_SIZE + 1];// page table directory
    ULONG               pageTableCount;// Number of actually used entries in page Tables.
    MMU_CONFIG          hwMmuConfig;// Preinitialized hardware mmu config for application

} APP_CTX, *PAPP_CTX;               // Client/application context

typedef struct _APP_CTX2
{
    WDFREQUEST  notifyRequest;          // request to be completed to notify user application after memory map
    PUINT64     notifyBuffer;           // notify request buffer will be filled with memory id
    UINT64      memoryIdCounter;
    WDFSPINLOCK memoryIdLock;
    LIST_ENTRY  memoryListHead;
    WDFSPINLOCK memoryListLock;

    APP_CTX     appCtx1;                // for backward compatibility
} APP_CTX2, *PAPP_CTX2;

/**
 * Getter for File context
 */
WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(APP_CTX2, GetFileContext)


/******************************************************************************
 * Device specific
 *****************************************************************************/

 /**
 * Device Memory pool tag name
 */
#define MEM_POOL_TAG ((ULONG)'ANGI')

/**
 * Max device/application buffer size
 */
#define HW_MAX_MEM_SIZE (PT_DIR_SIZE * PT_ENTRY_NO * PAGE_SIZE)

/**
 * Driver performance profiler
 */
typedef struct _PROFILER
{
    profiler_tsc        startHW;    // setting up and issuing HW scoring profiler
    profiler_tsc        scoreHW;    // HW scoring profiler
    profiler_tsc        intProc;    // interrupt profiler

} PROFILER, *PPROFILER;             // Driver performance profiler


/**
 * Device driver configuration
 */
typedef struct _DEV_CONFIG
{
    WDFDMAENABLER       dmaEnabler;     // DMA Enabler
    WDFINTERRUPT        interrupt;      // Interrupt
    BOOLEAN             interruptible;  // flag indicating if drv is currently servicing LEGACY interrupt
    BOOLEAN             d0i3Enabled;    // flag indicating if hw is in internal D0i3 power state
    WDFSPINLOCK         pwrLock;        // spinlock for protection of async power state access
    GNA_CPBLTS          cpblts;         // bitmap indicates device possibilities

} DEV_CONFIG, *PDEV_CONFIG;         // Device driver configuration

/**
 * Current request state
 */
typedef struct _REQ_STATE
{
    WDFREQUEST          req;        // request being currently processed and owned by drv
    PGNA_CALC_IN        data;       // active request data buffer address
    BOOLEAN             timeouted;  // flag indicating if hardware is in live-loop and timeout occurred
    BOOLEAN             hwVerify;   // flag indicating if request is for hardware verification purposes
    WDFSPINLOCK         reqLock;    // spinlock for protection of async req. state access

} REQ_STATE, *PREQ_STATE;           // Current request state

/**
 * Current App state
 */
typedef struct _APP_STATE
{
    PAPP_CTX2            app;        // file context of application from which last scoring was issued
    WDFSPINLOCK         appLock;    // spinlock for protection of async app state access

} APP_STATE, *PAPP_STATE;           // Current App state

/**
 * Device Hardware MMIO registers
 */
typedef struct _DEV_HW_REGS
{
    BUS_INTERFACE_STANDARD busInterface;// Device bus interface
    P_HW_REGS           regs;       // Memory-mapped hw registers
    ULONG               regsLength; // Length of the MMIO area

} DEV_HW, *PDEV_HW;                 // Hardware registers

/**
 * Device context
 */
typedef struct _DEV_CTX
{
    REQ_STATE           req;        // current request state context
    APP_STATE           app;        // current application state context
    DEV_CONFIG          cfg;        // driver configuration objects
    WDFTIMER            timeout;    // request timeout timer
    WDFQUEUE            queue;      // Device request queue
    WDFQUEUE            memoryMapQueue;// MemoryMap request queue
    DEV_HW              hw;         // hardware registers
    HW_DESC             desc;       // hardware descriptor (backward compatibility)
    PROFILER            profiler;   // profiler object for performance measurements

} DEV_CTX, *PDEV_CTX;               // Device context

/**
 * Device context getter
 */
WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEV_CTX, DeviceGetContext)


/**
 * Error check macro
 */
#define ERRCHECKP(condition, exitcode) if(condition){ return exitcode; }

#endif // #define _DRIVER_H
