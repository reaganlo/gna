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

#if !defined(_MEMORY2__H)
#define _MEMORY2__H

#include "Driver.h"

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
MemoryMap2(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx,
    _In_    PAPP_CTX2   appCtx,
    _In_    PMDL        pMdl,
    _In_    WDFREQUEST  mapRequest,
    _In_    UINT32      length);

/**
 * Unlocks the application memory buffer
 * and frees system objects associated with locked area.
 */
VOID
MemoryMapRelease2(
    _Inout_ PAPP_CTX2      appCtx,
    _Inout_ PMEMORY_CTX    memoryCtx);

/**
 * Finds memory context in the list
 * Should be called with acquired list lock
 *
 * @appCtx      app context
 * @memoryId    memory id
 * @return      memory context
 */

PMEMORY_CTX FindMemoryContextByIdLocked(
    _In_ PAPP_CTX2 appCtx,
    _In_ UINT64 memoryId);

#endif // _MEMORY__H
