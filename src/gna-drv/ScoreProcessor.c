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
#include "scoreprocessor.tmh"
#include "Memory.h"
#include "Hw.h"
#include "gna-etw-manifest.h"

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/

/**
 * Starts processing scoring request by hardware
 * NOTE:  ScoreStart is Write or HW INT driven, but is always called synchronously
 *
 * @devCtx              context of device
 * @request             input request
 *
 * @return  start status, on error completes request with error information
 */
static
NTSTATUS
ScoreStart(
    _In_    PDEV_CTX    devCtx,
    _In_    WDFREQUEST  request);

NTSTATUS
ScoreValidateParams(
    _In_    PGNA_CALC_IN config);

/**
 * Completes scoring request processed by hardware
 * NOTE:  ScoreFinalize is HW INT driven, always called synchronously
 *
 * @devCtx              context of device
 *
 * @return  start status, on error completes request with error information
 */
NTSTATUS
ScoreFinalize(
    _In_    PDEV_CTX    devCtx);

VOID
GNAScoreDebug(
    _In_ PDEV_CTX devCtx);

/******************************************************************************
 * Public Methods
 ******************************************************************************/

VOID
ScoreSubmitEvnt(
    WDFQUEUE            queue,
    WDFREQUEST          request,
    size_t              length)
{
    PDEV_CTX devCtx = WDF_NO_HANDLE;
    NTSTATUS status = STATUS_SUCCESS;

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(WdfIoQueueGetDevice(queue));

    // verify length is sufficient
    if (length < REQUEST_SIZE)
    {
        status = STATUS_BUFFER_TOO_SMALL;
        TraceFailMsg(TLE, T_EXIT, "Input data has invalid size", status);
        WdfRequestComplete(request, status);
    }
    else
    {
        ScoreStart(devCtx, request);
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
}

VOID
ScoreProcessorSleep(
    _In_    PDEV_CTX    devCtx)
{
    WDF_IO_QUEUE_STATE  state;

    TraceEntry(TLI, T_ENT);
    state = WdfIoQueueGetState(devCtx->queue, NULL, NULL);
    if (WDF_IO_QUEUE_IDLE(state))
    {
        Trace(TLV, T_QUE, "%!FUNC! Queue Idle. Put HW to D0i3 sleep.");
        HwPowerSwitch(devCtx->hw.regs, &devCtx->cfg, HW_POWER_OFF);
    }
    else
    {
        Trace(TLV, T_QUE, "%!FUNC! Queue not idle.");
    }
}

VOID
ScoreProcessorWakeup(
    _In_    PDEV_CTX    devCtx)
{
    TraceEntry(TLI, T_ENT);
    HwPowerSwitch(devCtx->hw.regs, &devCtx->cfg, POWER_ON);
}

VOID
ScoreComplete(
    _In_        PDEV_CTX    devCtx,
    _In_        NTSTATUS    status,
    _In_opt_    WDFREQUEST  request,
    _In_        BOOLEAN     hwUnmap,
    _In_opt_    PAPP_CTX    appCancel)
{
    TraceEntry(TLI, T_ENT);

    WdfSpinLockAcquire(devCtx->app.appLock);
    // if canceling app req. do not complete if appCancel is not active app
    if (NULL != appCancel && appCancel != devCtx->app.app)
    {
        Trace(TLI, T_QUE, "%!FUNC! Quiting - appCancel is not active.");
        WdfSpinLockRelease(devCtx->app.appLock);
        return;
    }
    // reset hw mapping
    if (hwUnmap)
    {
        //HwUnmapMemory(&modelCtx->desc.va->mmu_config);
        devCtx->app.app = NULL;
    }
    WdfSpinLockRelease(devCtx->app.appLock);
    HwSetInterruptible(&devCtx->cfg, FALSE);
    WdfSpinLockAcquire(devCtx->req.reqLock);
    // get request to complete
    if (WDF_NO_HANDLE == request)
    {
        request = devCtx->req.req;
    }
    // reset request state
    devCtx->req.req = WDF_NO_HANDLE;
    devCtx->req.data = NULL;
    devCtx->req.timeouted = FALSE;
    WdfSpinLockRelease(devCtx->req.reqLock);
    // complete request
    if (WDF_NO_HANDLE != request)
    {
        Trace(TLI, T_QUE, "%!FUNC! Completing current request.");
        WdfRequestComplete(request, status);
    }
}

VOID
ScoreCancelReqByApp(
    _In_    WDFQUEUE        queue,
    _In_    WDFFILEOBJECT   app)
{
    NTSTATUS   status = STATUS_SUCCESS;
    WDFREQUEST cancelReq = WDF_NO_HANDLE;   // handle of retrieved request

    Trace(TLI, T_QUE, "%!FUNC! Cancel requests from app: (%p)", app);
    // cancel pending requests first
    status = WdfIoQueueRetrieveRequestByFileObject(queue, app, &cancelReq);
    while (STATUS_SUCCESS == status)
    {
        Trace(TLW, T_QUE, "%!FUNC! Cancel pending request %p.", cancelReq);
        WdfRequestComplete(cancelReq, STATUS_CANCELLED);

        // get next request if any left
        status = WdfIoQueueRetrieveRequestByFileObject(queue, app, &cancelReq);
    }
}

BOOLEAN
InterruptIsrEvnt(
    WDFINTERRUPT        interrupt,
    ULONG               messageID)
{
    PDEV_CTX devCtx;
#ifdef ENABLE_LEGACY_INTERRUPTS
    ULONG32         stsReg;      // HW status register value
#endif // ENABLE_LEGACY_INTERRUPTS
    UNREFERENCED_PARAMETER(messageID);

    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(WdfInterruptGetDevice(interrupt));

#ifdef ENABLE_LEGACY_INTERRUPTS
    if (FALSE == devCtx->cfg.interruptible)
    {
        return FALSE;
    }

    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    // Read hw STS and test if any hw interrupt flag is set
    stsReg = HwReadReg(&devCtx->hw.regs->sts, 0);
    if (!(HW_INT_FLAGS & stsReg))
    {
        return FALSE;
    }
    devCtx->cfg.interruptible = FALSE;
#endif // ENABLE_LEGACY_INTERRUPTS

    TraceEntry(TLI, T_ENT);
    WdfInterruptQueueDpcForIsr(devCtx->cfg.interrupt);

    EventWriteDriverApiEnd(NULL, __FUNCTION__);

    return TRUE;
}

VOID
InterruptDpcEvnt(
    WDFINTERRUPT        interrupt,
    WDFOBJECT           dev)
{
    NTSTATUS         status = STATUS_SUCCESS;
    PDEV_CTX  devCtx;         // device context
    UNREFERENCED_PARAMETER(interrupt);

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    devCtx = DeviceGetContext((WDFDEVICE)dev);

    // check active request to complete
    WdfSpinLockAcquire(devCtx->req.reqLock);
    if (WDF_NO_HANDLE == devCtx->req.req) status = STATUS_REQUEST_OUT_OF_SEQUENCE;
    WdfSpinLockRelease(devCtx->req.reqLock);
    if (!NT_SUCCESS(status))  // no active request?
    {
        TraceFailMsg(TLE, T_EXIT, "Active request not set.", status);
        EventWriteDriverApiEnd(NULL, __FUNCTION__);
        return; // nothing to do, unknown state
    }

    // complete processed request, ignore errors, as request is always completed
    WdfTimerStop(devCtx->timeout, WdfFalse);
    status = ScoreFinalize(devCtx);
    if (STATUS_MORE_PROCESSING_REQUIRED == status)
    {
        Trace(TLI, T_INIT, "%!FUNC!: More processing required");
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
}

VOID
ScoreTimeoutEvnt(
    WDFTIMER            timer)
{
    WDFDEVICE dev;
    PDEV_CTX devCtx;
    CTRL_REG ctrl;

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    dev = WdfTimerGetParentObject(timer);
    devCtx = DeviceGetContext(dev);
    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    ctrl._dword = HwReadReg(&devCtx->hw.regs->ctrl, 0);
    if (1 == ctrl.start_accel)
    {
        WdfSpinLockAcquire(devCtx->req.reqLock);
        devCtx->req.timeouted = TRUE;
        WdfSpinLockRelease(devCtx->req.reqLock);
        HwAbort(devCtx->hw.regs);
/**
 * Rationale for warning suppression:
 * Argument interrupt (parameter 1) is ignored by function InterruptDpcEvnt
 */
#pragma warning(suppress: 6387)
        InterruptDpcEvnt(WDF_NO_HANDLE, dev);
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
}

/******************************************************************************
 * Private Methods
 ******************************************************************************/

VOID
GNAScoreDebug(
    _In_ PDEV_CTX devCtx)
{
    ULONG read32 = 0;
    P_HW_REGS regs = devCtx->hw.regs;

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x80);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA labase: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA lacnt: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x82);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA {nInputElements, nnFlags, nnop}: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA {inputIteration/nInputConvStride,grouping,nOutputElements}: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x83);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA {res,outFbIter,inputInLastIter/nConvFilterSize}: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA {outFbInLastIter/poolStride,outFbInFirstIter/nConvFilters}: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x84);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA {nActListElements/nCopyElements,res,nActivationSegments/poolSize}: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA reserved: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x86);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA inArrayPtr: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA outArrayPtr: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x87);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA outArraySumPtr: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA outFbArrayActPtr: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x88);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA {weightArrayPtr/filterArrayPtr}: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA constArrayPtr: %#x:", read32);

    WRITE_REGISTER_ULONG((volatile ULONG*)&regs->isi._dword, 0x89);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.low);
    Trace(TLI, T_MEM, "GNA actOutputListPtr: %#x:", read32);
    read32 = READ_REGISTER_ULONG((volatile ULONG*)&regs->isv.hi);
    Trace(TLI, T_MEM, "GNA actFuncSectDefPtr: %#x:", read32);
}

static size_t calculateLayersDescriptorBufferSize(const PGNA_CALC_IN input)
{
    const max_ = max(sizeof(XNN_ACTIVE_LIST_DESCR), sizeof(GMM_ACTIVE_LIST_DESCR));

    size_t sz = sizeof(GNA_CALC_IN);
    sz += sizeof(GNA_BUFFER_DESCR)*input->reqCfgDescr.buffersCount;        
    sz += sizeof(NNOP_TYPE_DESCR)*input->reqCfgDescr.nnopTypesCount;
    sz += max_*input->reqCfgDescr.xnnActiveListsCount;
    sz += max_*input->reqCfgDescr.gmmActiveListsCount;

    return sz;
}

static void setLayersDescriptorParameters(const PGNA_CALC_IN input, PMODEL_CTX modelCtx)
{
    if (input->reqCfgDescr.requestConfigId == modelCtx->requestConfigId)
    {
        return;
    }
    modelCtx->requestConfigId = input->reqCfgDescr.requestConfigId;

    PUCHAR const memoryBase = modelCtx->userMemoryBaseVA;

    // set buffers according to request config
    PGNA_BUFFER_DESCR bufferDescr = (PGNA_BUFFER_DESCR)((PUCHAR)input + sizeof(GNA_CALC_IN));
    for (UINT32 i = 0; i < input->reqCfgDescr.buffersCount; ++i)
    {
        *(PUINT32)(memoryBase + bufferDescr->offset) = bufferDescr->value;
        ++bufferDescr;
    }

    // set nnop type
    PNNOP_TYPE_DESCR nnopTypeDescr = (PNNOP_TYPE_DESCR)bufferDescr;
    for (UINT32 i = 0; i < input->reqCfgDescr.nnopTypesCount; ++i)
    {
        *(PUINT8)(memoryBase + nnopTypeDescr->offset) = nnopTypeDescr->value;
        ++nnopTypeDescr;
    }

    // set xnn active list params according to request config
    PXNN_ACTIVE_LIST_DESCR xnnActLstDescr = (PXNN_ACTIVE_LIST_DESCR)nnopTypeDescr;
    for (UINT32 i = 0; i < input->reqCfgDescr.xnnActiveListsCount; ++i)
    {
        *(PUINT32)(memoryBase + xnnActLstDescr->act_list_buffer_offset) = xnnActLstDescr->act_list_buffer_value;
        *(PUINT16)(memoryBase + xnnActLstDescr->act_list_n_elems_offset) = xnnActLstDescr->act_list_n_elems_value;
        ++xnnActLstDescr;
    }

    // set gmm active list params according to request config
    PGMM_ACTIVE_LIST_DESCR gmmActLstDescr = (PGMM_ACTIVE_LIST_DESCR)xnnActLstDescr;
    for (UINT32 i = 0; i < input->reqCfgDescr.gmmActiveListsCount; ++i)
    {
        *(PUINT32)(memoryBase + gmmActLstDescr->asladdr_offset) = gmmActLstDescr->asladdr_value;
        *(PUINT32)(memoryBase + gmmActLstDescr->astlistlen_offset) = gmmActLstDescr->astlistlen_value;
        *(PUINT32)(memoryBase + gmmActLstDescr->gmmscrlen_offset) = gmmActLstDescr->gmmscrlen_value;
        ++gmmActLstDescr;
    }
}

NTSTATUS
ScoreStart(
    _In_    PDEV_CTX    devCtx,
    _In_    WDFREQUEST  request)
{
    NTSTATUS    status = STATUS_INVALID_DEVICE_REQUEST;
    PAPP_CTX    appCtx = NULL; // file context of device for calling application
    PMODEL_CTX  modelCtx = NULL; // current model context
    BOOLEAN     isAppCurrent = FALSE;// deterimenes if current app is saved as recent
    size_t      inputLength = 0;    // tmp in request buffer length
    PGNA_CALC_IN input = NULL; // input parameters
    PVOID       lyrDscBuffer = NULL; // layer descriptor buffer address to save config to

    TraceEntry(TLI, T_ENT);

    profilerDTscStart(&devCtx->profiler.startHW);
    // get app context and verify if has mem mapped
    appCtx = GetFileContext(WdfRequestGetFileObject(request));
    // get pointer to request data and verify if valid
    status = WdfRequestRetrieveInputBuffer(request, REQUEST_SIZE, &input, &inputLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        goto cleanup;
    }

    if (calculateLayersDescriptorBufferSize(input) != inputLength)
    {
        TraceFailMsg(TLE, T_EXIT, "Score input buffer wrong size", STATUS_INVALID_BUFFER_SIZE);
        goto cleanup;
    }

    status = ScoreValidateParams(input);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "(Score parameters are invalid)", status);
        goto cleanup;
    }
    modelCtx = appCtx->models[input->modelId];
    if (NULL == modelCtx)
    {
        TraceFailMsg(TLI, T_MEM, "Application has NOT mapped memory!", status);
        goto cleanup;
    }

    setLayersDescriptorParameters(input, modelCtx);

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

    // setup timer for recovery after DRV_RECOVERY_TIMEOUT seconds from now
    if (WdfTrue == WdfTimerStart(devCtx->timeout, -10000000 * DRV_RECOVERY_TIMEOUT))
    {
        status = STATUS_TIMER_NOT_CANCELED;
        goto cleanup;
    }
    // configure device for scoring and start, parameters already validated
    lyrDscBuffer = MmGetSystemAddressForMdlSafe(modelCtx->pMdl, HighPagePriority | MdlMappingNoExecute);
    if (NULL == lyrDscBuffer)
    {
        status = STATUS_INVALID_ADDRESS;
        goto cleanup;
    }
    HwInitExecution(devCtx->hw.regs, (ULONG)modelCtx->desc.la.QuadPart, &modelCtx->desc.va->xnn_config, input, &devCtx->cfg);

    profilerDTscStop(&devCtx->profiler.startHW);
    profilerTscStart(&devCtx->profiler.scoreHW);
    Trace(TLV, T_QUE, "%!FUNC!: Scoring started, startHW time %llu", devCtx->profiler.startHW.passed);
    return status; // SUCCESS - request will be completed by interrupt

cleanup: // ERROR - complete request
    if (!NT_SUCCESS(status))
    {
        ScoreComplete(devCtx, status, request, TRUE, NULL);
        ScoreProcessorSleep(devCtx);
        TraceFail(TLE, T_EXIT, status);
    }
    return status;
}

NTSTATUS
ScoreValidateParams(
    _In_    PGNA_CALC_IN    params)
{
    TraceEntry(TLI, T_ENT);

    ERRCHECKP(NULL == params, STATUS_DATA_ERROR);
    ERRCHECKP(params->ctrlFlags.gnaMode > 1, STATUS_INVALID_PARAMETER);

    return STATUS_SUCCESS;
}

NTSTATUS
ScoreFinalize(
    _In_    PDEV_CTX    devCtx)
{
    PGNA_CALC_IN        output;     // request output parameters
    status_t            hwSts;      // hardware status
    PROFILE_D_(ULONG32  ptcReg = 0);// HW Performance Total Cycle register value
    PROFILE_D_(ULONG32  pscReg = 0);// HW Performance Stall Cycle register value

    TraceEntry(TLI, T_ENT);

    profilerDTscStart(&devCtx->profiler.intProc);
    // stop HW scoring profiler on completion note, when scoring is not completed
    // (breakpoint/error interrupt) profiler time is invalid
    profilerTscStop(&devCtx->profiler.scoreHW);
    hwSts = HwGetIntStatus(devCtx, devCtx->hw.regs);
    // if no status change -> warning or unsupported int return, do not complete request
    if (GNA_DEVICEBUSY == hwSts) return STATUS_MORE_PROCESSING_REQUIRED;

    // set status to success and re-set to error if any occured
    output = devCtx->req.data;
    RtlZeroMemory(&output->drvPerf, sizeof(perf_drv_t));
    RtlZeroMemory(&output->hwPerf, sizeof(perf_hw_t));
    output->status = hwSts; // save hardware status

    // read and save performance counters
#if defined(PROFILE)
    output->drvPerf.scoreHW = devCtx->profiler.scoreHW.passed;
    Trace(TLI, T_QUE, "%!FUNC!: scoreHW time %llu", output->drvPerf.scoreHW);
    EventWriteScoreCycles(NULL, output->drvPerf.scoreHW);
#if defined(PROFILE_DETAILED)
    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    ptcReg = HwReadReg(&devCtx->hw.regs->ptc, 0);
    pscReg = HwReadReg(&devCtx->hw.regs->psc, 0);
    output->hwPerf.stall = (time_tsc)pscReg;
    if (PC_REG_SATURATED == pscReg)
    {
        output->hwPerf.stall = TIME_TSC_MAX;
    }
    output->hwPerf.total = (time_tsc)ptcReg;
    if (PC_REG_SATURATED == ptcReg)
    {
        output->hwPerf.total = TIME_TSC_MAX;
    }
    output->drvPerf.startHW = devCtx->profiler.startHW.passed;
    profilerDTscStop(&devCtx->profiler.intProc);
    output->drvPerf.intProc = devCtx->profiler.intProc.passed;
    // print performance results
    Trace(TLI, T_QUE, "%!FUNC!: startHW time %llu", output->drvPerf.startHW);
    Trace(TLI, T_QUE, "%!FUNC!: intProc time %llu", output->drvPerf.intProc);
    Trace(TLI, T_QUE, "%!FUNC!: stall time %llu", output->hwPerf.stall);
    Trace(TLI, T_QUE, "%!FUNC!: total time %llu", output->hwPerf.total);
#endif // PROFILE_DETAILED
#endif // PROFILE

    GNAScoreDebug(devCtx);

    ScoreComplete(devCtx, STATUS_SUCCESS, WDF_NO_HANDLE, FALSE, NULL);
#ifndef _DEBUG  //  do not poweroff in debug mode to keep MMIO status regs
    ScoreProcessorSleep(devCtx);
#endif // !_DEBUG
    return STATUS_SUCCESS;
}

VOID
ScoreDeferredUnmap(
    _In_    PDEV_CTX    devCtx,
    _In_    WDFREQUEST  unmapReq)
{
    NTSTATUS status = STATUS_SUCCESS;
    WDFFILEOBJECT app = NULL;     // file object of calling application
    PAPP_CTX      appCtx = NULL;     // file context of calling application

    TraceEntry(TLI, T_ENT);
    // cancel all request from current application
    app = WdfRequestGetFileObject(unmapReq);
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

    PUINT64 mid;
    size_t midLength;

    // retrieve and store input params
    status = WdfRequestRetrieveInputBuffer(unmapReq, sizeof(UINT64), &mid, &midLength);
    Trace(TLI, T_MEM, "Model id sent from userland: %llu", *mid);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        goto ioctl_mm_error;
    }

    // not compatible data sent from userland
    if (sizeof(*mid) != midLength)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Bad data sent", status);
        goto ioctl_mm_error;
    }

    // bad model id
    if (*mid >= APP_MODELS_LIMIT)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Bad model id", status);
        goto ioctl_mm_error;
    }

    // perform unmapping
    PMODEL_CTX modelCtx = appCtx->models[*mid];
    if (NULL == modelCtx)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "No model context for given model id", status);
        goto ioctl_mm_error;
    }
    MemoryMapRelease(appCtx, modelCtx);

    // complete unmap request
ioctl_mm_error:
    WdfRequestComplete(unmapReq, status);
}
