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

#include "ScoreProcessor.h"
#include "ScoreProcessor2.tmh"
#include "Memory.h"
#include "Memory2.h"
#include "Hw.h"
#include "gna-etw-manifest.h"

#define ALIGN(number, significance)   (((int)((number) + significance -1) / significance) * significance)

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/

/**
 * Gets the next patch in request structure
 * @patch current patch
 * @return next patch structure
 */
static
PGNA_MEMORY_PATCH
GetNextPatch(
    _In_ PGNA_MEMORY_PATCH patch);

/**
 * Patches user memory
 * @input request structure
 * @memoryCtx memory context used for patching memory
 */
static
VOID
PatchMemory(
    _In_ const PGNA_CALC_IN input,
    _In_ const PMEMORY_CTX memoryCtx);

/**
 * Validates user request
 * @input request structure
 * @inputLength request structure size
 * @appCtx application context
 */
static
NTSTATUS
ScoreValidateParams2(
    _In_    PGNA_CALC_IN input,
    _In_    size_t       inputLength,
    _In_    PAPP_CTX2    appCtx);

/******************************************************************************
 * Private Methods
 ******************************************************************************/

PGNA_MEMORY_PATCH
GetNextPatch(
    _In_ PGNA_MEMORY_PATCH patch)
{
    return (PGNA_MEMORY_PATCH)(patch->data + patch->size);
}

VOID
PatchMemory(
    _In_ const PGNA_CALC_IN input,
    _In_ const PMEMORY_CTX memoryCtx)
{
    PUCHAR const memoryBase = memoryCtx->userMemoryBaseVA;

    PGNA_MEMORY_PATCH memoryPatch;
    PUINT8 memoryDestination;
    UINT32 i;

    /* update memory according to request config */
    Trace(TLV, T_MEM, "patchCount %llu\n", input->patchCount);

    memoryPatch = (PGNA_MEMORY_PATCH)input->patches;
    for (i = 0; i < input->patchCount; ++i)
    {
        Trace(TLV, T_MEM, "patch %d. offset: %llu, size: %llu\n",
            i, memoryPatch->offset, memoryPatch->size);

        memoryDestination = memoryBase + memoryPatch->offset;
        RtlCopyMemory(memoryDestination, memoryPatch->data, memoryPatch->size);
        memoryPatch = GetNextPatch(memoryPatch);
    }
}

NTSTATUS
ScoreValidateParams2(
    _In_    PGNA_CALC_IN    input,
    _In_    size_t          inputLength,
    _In_    PAPP_CTX2       appCtx)
{
    PMEMORY_CTX memoryCtx = NULL;
    PGNA_MEMORY_PATCH memoryPatch = NULL;
    size_t calculatedRequestSize;
    size_t userRequestSize;
    UINT64 patchBoundary;
    int i;

    TraceEntry(TLI, T_ENT);

    ERRCHECKP(NULL == input, STATUS_DATA_ERROR);
    ERRCHECKP(input->ctrlFlags.gnaMode > 1, STATUS_INVALID_PARAMETER);
    ERRCHECKP(input->configSize != inputLength, STATUS_INVALID_PARAMETER);

    memoryCtx = FindMemoryContextByIdLocked(appCtx, input->memoryId);
    calculatedRequestSize = sizeof(GNA_CALC_IN);
    userRequestSize = input->configSize;
    memoryPatch = (PGNA_MEMORY_PATCH)(
        (PUINT8)input + sizeof(GNA_CALC_IN));

    for (i = 0; i < input->patchCount; ++i)
    {
        // validate if memory patch exceeds memory boundary
        patchBoundary = memoryPatch->offset + memoryPatch->size;
        if (patchBoundary > memoryCtx->userMemorySize) {
            Trace(TLE, T_MEM, "patch %d exceeds memory boundary", i);
            return STATUS_INVALID_PARAMETER;
        }

        // validate if calculated size exceeds size provide by user
        calculatedRequestSize += sizeof(GNA_MEMORY_PATCH)
            + memoryPatch->size;

        if (calculatedRequestSize > userRequestSize) {
            Trace(TLE, T_MEM, "patch %d exceeds user request size\n", i);
            return STATUS_INVALID_PARAMETER;
        }

        memoryPatch = GetNextPatch(memoryPatch);
    }

    calculatedRequestSize = ALIGN(calculatedRequestSize, sizeof(UINT64));
    ERRCHECKP(calculatedRequestSize != inputLength, STATUS_INVALID_PARAMETER);


    return STATUS_SUCCESS;
}

VOID
ScoreDeferredUnmap2(
    _In_    PDEV_CTX    devCtx,
    _In_    WDFREQUEST  unmapReq)
{
    NTSTATUS status = STATUS_SUCCESS;
    WDFFILEOBJECT app = NULL;     // file object of calling application
    PAPP_CTX2      appCtx = NULL;     // file context of calling application
    PMEMORY_CTX   memoryCtx = NULL;

    TraceEntry(TLI, T_ENT);
    // cancel all request from current application
    app = WdfRequestGetFileObject(unmapReq);

    // FIXME: cancel requests by memory
    ScoreCancelReqByApp(devCtx->queue, app);
    Trace(TLV, T_QUE, "%!FUNC! Force Powering off HW to D0i3.");
    HwPowerSwitch(devCtx->hw.regs, &devCtx->cfg, HW_POWER_OFF);

    // clean currently active app flag
    appCtx = GetFileContext(app);
    WdfSpinLockAcquire(devCtx->app.appLock);
    if (appCtx == devCtx->app.app)
    {
        devCtx->app.app = NULL;
    }
    WdfSpinLockRelease(devCtx->app.appLock);

    appCtx = GetFileContext(WdfRequestGetFileObject(unmapReq));

    PUINT64 memoryId;
    size_t memoryIdLength;

    // retrieve and store input params
    status = WdfRequestRetrieveInputBuffer(unmapReq, sizeof(*memoryId), &memoryId, &memoryIdLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        goto ioctl_mm_error;
    }

    // not compatible data sent from userland
    if (sizeof(*memoryId) != memoryIdLength)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Bad data sent", status);
        goto ioctl_mm_error;
    }

    Trace(TLI, T_MEM, "Model id sent from userland: %llu", *memoryId);

    // perform unmapping
    memoryCtx = FindMemoryContextByIdLocked(appCtx, *memoryId);
    if (NULL == memoryCtx)
    {
        status = GNA_ERR_MEMORY_ALREADY_UNMAPPED;
        TraceFailMsg(TLE, T_EXIT, "No memory context for given memory id", status);
        goto ioctl_mm_error;
    }
    MemoryMapRelease2(appCtx, memoryCtx);

    // complete unmap request
ioctl_mm_error:
    WdfRequestComplete(unmapReq, status);
}


/******************************************************************************
 * Public Methods
 ******************************************************************************/

NTSTATUS
ScoreStart2(
    _In_ PGNA_CALC_IN input,
    _In_ size_t inputLength,
    _In_ PAPP_CTX2 appCtx,
    _In_ PDEV_CTX devCtx,
    _In_ WDFREQUEST request)
{
    NTSTATUS status = STATUS_INVALID_DEVICE_REQUEST;
    PMEMORY_CTX memoryCtx = NULL; // current memory context
    BOOLEAN isAppCurrent = FALSE;// deterimenes if current app is saved as recent
    PVOID lyrDscBuffer = NULL; // layer descriptor buffer address to save config to

    TraceEntry(TLI, T_ENT);

    profilerDTscStart(&devCtx->profiler.startHW);
    // get app context and verify if has mem mapped
    appCtx = GetFileContext(WdfRequestGetFileObject(request));
    // get pointer to request data and verify if valid
    status = WdfRequestRetrieveInputBuffer(request, REQUEST_SIZE, &input, &inputLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        return status;
    }

    status = ScoreValidateParams2(input, inputLength, appCtx);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "(Score parameters are invalid)", status);
        return status;
    }

    // check and remember application from which req. is being processed now
    WdfSpinLockAcquire(devCtx->req.reqLock);
    devCtx->req.req = request;
    devCtx->req.data = input;
    devCtx->req.timeouted = FALSE;
    WdfSpinLockRelease(devCtx->req.reqLock);
    WdfSpinLockAcquire(devCtx->app.appLock);
    isAppCurrent = devCtx->app.app == appCtx;
    devCtx->app.app = appCtx;
    WdfSpinLockRelease(devCtx->app.appLock);

    memoryCtx = FindMemoryContextByIdLocked(appCtx, input->memoryId);
    if (NULL == memoryCtx)
    {
        status = STATUS_INVALID_PARAMETER;
        TraceFailMsg(TLI, T_MEM, "Application has NOT mapped memory!", status);
        return status;
    }

    PatchMemory(input, memoryCtx);

    // setup timer for recovery after DRV_RECOVERY_TIMEOUT seconds from now
    if (WdfTrue == WdfTimerStart(devCtx->timeout, -10000000LL * devCtx->cfg.cpblts.recoveryTimeout))
    {
        status = STATUS_TIMER_NOT_CANCELED;
        return status;
    }

    // configure device for scoring and start, parameters already validated
    lyrDscBuffer = MmGetSystemAddressForMdlSafe(memoryCtx->pMdl, HighPagePriority | MdlMappingNoExecute);
    if (NULL == lyrDscBuffer)
    {
        status = STATUS_INVALID_ADDRESS;
        return status;
    }

    HwInitExecution(devCtx->hw.regs, &memoryCtx->desc.va->xnn_config, lyrDscBuffer,
        input, (ULONG)memoryCtx->desc.la.QuadPart, &devCtx->cfg);

    profilerDTscStop(&devCtx->profiler.startHW);
    profilerTscStart(&devCtx->profiler.scoreHW);
    Trace(TLV, T_QUE, "%!FUNC!: Scoring started, startHW time %llu", profilerGetTscPassed(&devCtx->profiler.startHW));
    return status; // SUCCESS - request will be completed by interrupt
}
