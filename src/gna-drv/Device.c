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

#include "Device.h"
#include "device.tmh"
#include "ScoreProcessor.h"
#include "Memory.h"
#include "gna-etw-manifest.h"

/******************************************************************************
 * Public Methods
 ******************************************************************************/

NTSTATUS
DeviceD0EntryEvnt(
    WDFDEVICE              dev,
    WDF_POWER_DEVICE_STATE PreviousState)
{
    PDEV_CTX devCtx = NULL;
    UNREFERENCED_PARAMETER(PreviousState);

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);
    EventWriteDeviceD0Entry(NULL);

    // power off hw to save power till first score arrives
    devCtx = DeviceGetContext(dev);
    ScoreProcessorSleep(devCtx);

    EventWriteDriverApiEnd(NULL, __FUNCTION__);

    return STATUS_SUCCESS;
}

NTSTATUS
DeviceD0ExitEvnt(
    WDFDEVICE              dev,
    WDF_POWER_DEVICE_STATE targetState)
{
    PDEV_CTX devCtx;

    UNREFERENCED_PARAMETER(targetState);

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);
    EventWriteDeviceD0Exit(NULL);

    devCtx = DeviceGetContext(dev);
    // power on hw to enable correct D3 device transition
    ScoreProcessorWakeup(devCtx);

    // reset state
    ScoreComplete(devCtx, STATUS_CANCELLED, WDF_NO_HANDLE, TRUE, NULL);

    // clean both queues
    WdfIoQueuePurgeSynchronously(devCtx->queue);
    WdfIoQueuePurgeSynchronously(WdfDeviceGetDefaultQueue(dev));

    // resume queues to function normally when D0 is entering
    Trace(TLI, T_QUE, "%!FUNC! Queues purge completed, Both QUEUE START");
    WdfIoQueueStart(devCtx->queue);
    WdfIoQueueStart(WdfDeviceGetDefaultQueue(dev));

    EventWriteDriverApiEnd(NULL, __FUNCTION__);

    return STATUS_SUCCESS;
}


VOID
FileCreateEvnt(
    WDFDEVICE           dev,
    WDFREQUEST          request,
    WDFFILEOBJECT       appObj)
{
    PAPP_CTX appCtx     = NULL;
    PDEV_CTX devCtx     = NULL;

    UNREFERENCED_PARAMETER(dev);

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    appCtx = GetFileContext(appObj);
    RtlZeroMemory(appCtx, sizeof(APP_CTX));
    devCtx  = DeviceGetContext(dev);

    WdfRequestComplete(request, STATUS_SUCCESS);

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
}

VOID
FileCloseEvnt(
    WDFFILEOBJECT       appObj)
{
    PDEV_CTX devCtx     = NULL;
    PAPP_CTX appCtx     = NULL;

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(WdfFileObjectGetDevice(appObj));
    appCtx = GetFileContext(appObj);
    WdfSpinLockAcquire(devCtx->app.appLock);
    if (devCtx->app.app == appCtx) // app is current
    {
        Trace(TLW, T_QUE, "%!FUNC!: Application context RESET");
        devCtx->app.app = NULL;
    }
    WdfSpinLockRelease(devCtx->app.appLock);
    RtlZeroMemory(appCtx, sizeof(APP_CTX));

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
}

VOID
FileCleanupEvnt(
    WDFFILEOBJECT       appObj)
{
    PDEV_CTX devCtx     = NULL;

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(WdfFileObjectGetDevice(appObj));
    // cancel active request
    ScoreComplete(devCtx, STATUS_CANCELLED, WDF_NO_HANDLE, TRUE, GetFileContext(appObj));
    // power off hw if queue idle
    ScoreProcessorSleep(devCtx);
    // free resources but do not abort and clean hw regs
    // memory is unlocked regardless the status of unmap operation.

    PAPP_CTX appCtx = GetFileContext(appObj);
    UINT32 i;
    for (i = 0; i < APP_MODELS_LIMIT; i++)
    {
        PMODEL_CTX modelCtx = appCtx->models[i];
        if (NULL != modelCtx)
        {
            MemoryMapRelease(appCtx, modelCtx);
        }
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
}
