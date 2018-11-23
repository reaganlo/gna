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

#if !defined(_MEMORY__H)
#define _MEMORY__H

#include "Driver.h"

/**
 * Old memory map interface IOCTL definitions
 */
#define GNA_IOCTL_MEM_MAP   CTL_CODE(FILE_DEVICE_PCI_GNA, 0x900, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_MEM_UNMAP CTL_CODE(FILE_DEVICE_PCI_GNA, 0x901, METHOD_BUFFERED, FILE_ANY_ACCESS)

/**
  * MEM_MAP IOCTL - input data.
  */
typedef struct _GNA_MM_IN
{
    void* POINTER_64 memoryBase;   // Base address of the application buffer
    UINT64 length;     // Length of the application buffer

} GNA_MM_IN, *PGNA_MM_IN;           // MEM_MAP IOCTL - input data

static_assert(16 == sizeof(GNA_MM_IN), "Invalid size of GNA_MM_IN");


/**
 * MEM_MAP IOCTL - output data.
 * Size:    266 776 B
 */
typedef struct _GNA_MM_OUT
{
    UINT8               reserved[266768]; // backward compatibility: do NOT use
    status_t            status;     // status of memory map
    UINT32              inBuffSize; // gna internal input buffer size

} GNA_MM_OUT, *PGNA_MM_OUT;         // MEM_MAP IOCTL - output data

static_assert(266776 == sizeof(GNA_MM_OUT), "Invalid size of GNA_MM_OUT");

/**
 * Performs user memory mapping
 *
 * @dev         device object
 * @devCtx      device context
 * @appCtx      app context
 * @usrBuffer   Base address of the application buffer
 * @length      Length of the application buffer
 * @outData     mapping output data to return
 * @return  mapping status
 */
NTSTATUS
MemoryMap(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx,
    _In_    PAPP_CTX    appCtx,
    _In_    void* POINTER_64 usrBuffer,
    _In_    UINT32      length,
    _Inout_ PGNA_MM_OUT outData);

/**
 * Unlocks the application memory buffer
 * and frees system objects associated with locked area.
 */
VOID
MemoryMapRelease(
    _Inout_ PAPP_CTX    appCtx);

NTSTATUS
ModelDescInit(
    _In_ PDEV_CTX     devCtx,
    _In_ PMEMORY_CTX   memoryCtx);
VOID
ModelDescRelease(
    _In_    PHW_DESC    desc);

status_t
CheckMapConfigParameters(
    _In_    PVOID     usrBuffer,
    _In_    UINT32    length);

#define     DIV_CEIL(x, y)          (((x)+(y)-1)/(y))

//NOTE: This is just a "dummy" subroutine, but it is necessary to initialize the fake DMA operation
DRIVER_LIST_CONTROL
ProcessSGList;

#ifdef ALLOC_PRAGMA
#pragma alloc_text (PAGE, ModelDescInit)
#endif

#endif // _MEMORY__H

