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

#include "IoControl.h"
#include "IoControl.tmh"
#include "Memory.h"
#include "ScoreProcessor.h"
#include "Hw.h"
#include "gna-etw-manifest.h"

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/

/**
 * Performs user memory mapping
 *
 * @dev                 device object handle
 * @devCtx              device context
 * @request             ioctl request
 */
static VOID
IoctlMemMap(
    _In_    WDFDEVICE   dev,
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request);

/**
 * Unmaps user mapped memory
 *
 * @devCtx              device context
 * @request             ioctl request
 */
static VOID
IoctlMemUnmap(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request);

/**
 * Stores request for later notification
 * @devCtx              device context
 * @request             ioctl request
 */
static VOID
IoctlNotify(
    _Inout_ WDFREQUEST  request);

/**
 * Reads device capabilities from device config
 * @devCtx              device context
 * @request             ioctl request
 */
static VOID
IoctlGetCapabilities(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request);

#ifdef DRV_DEBUG_INTERFACE
static VOID
IoctlReadReg(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request);

static VOID
IoctlWriteReg(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request);

static VOID
IoctlReadPageDir(
    _Inout_ WDFREQUEST  request);
#endif

/******************************************************************************
 * Public Methods
 ******************************************************************************/

VOID
IoctlDispatcher(
    WDFQUEUE            queue,
    WDFREQUEST          request,
    size_t              OutputBufferLength,
    size_t              InputBufferLength,
    ULONG               IoControlCode)
{
    NTSTATUS    status  = STATUS_SUCCESS;
    WDFDEVICE   dev;
    PDEV_CTX    devCtx  = NULL;
    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(InputBufferLength);

    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    TraceEntry(TLV, T_ENT);
    dev = WdfIoQueueGetDevice(queue);
    devCtx = DeviceGetContext(dev);
    if (NULL == devCtx)
    {
        status = STATUS_DATA_ERROR;
        TraceFailMsg(TLE, T_EXIT, "DeviceGetContext", status);
        WdfRequestComplete(request, status);
        return;
    }

    switch (IoControlCode)
    {
    case GNA_IOCTL_MEM_MAP:
        IoctlMemMap(dev, devCtx, request);
        break;

    case GNA_IOCTL_MEM_UNMAP:
        IoctlMemUnmap(devCtx, request);
        break;

    case GNA_IOCTL_WAKEUP_HW: // empty ioctl only to wake up device, complete immediately
        Trace(TLI, T_QUE, "%!FUNC! GNA_IOCTL_WAKEUP_HW");
        WdfRequestComplete(request, status); 
        break;

    case GNA_IOCTL_CPBLTS:
        IoctlGetCapabilities(devCtx, request);
        break;

    case GNA_IOCTL_NOTIFY:
        IoctlNotify(request);
        break;

#ifdef DRV_DEBUG_INTERFACE
    case GNA_IOCTL_READ_REG:
        IoctlReadReg(devCtx, request);
        break;
    
    case GNA_IOCTL_WRITE_REG:
        IoctlWriteReg(devCtx, request);
        break;

    case GNA_IOCTL_READ_PGDIR:
        IoctlReadPageDir(request);
        break;
#endif // DRV_DEBUG_INTERFACE

    default:
        status = STATUS_INVALID_DEVICE_REQUEST;
        TraceFailMsg(TLE, T_EXIT, "(Unknown IOCTL)", status);
        WdfRequestComplete(request, status);
        break;
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);

    return;
}

VOID
IoctlDeferred(
    WDFQUEUE            queue,
    WDFREQUEST          request,
    size_t              OutputBufferLength,
    size_t              InputBufferLength,
    ULONG               IoControlCode)
{
    NTSTATUS    status = STATUS_SUCCESS;
    WDFDEVICE   dev;
    PDEV_CTX    devCtx = NULL;

    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(InputBufferLength);

    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    TraceEntry(TLV, T_ENT);
    dev = WdfIoQueueGetDevice(queue);
    devCtx = DeviceGetContext(dev);
    if (NULL == devCtx)
    {
        status = STATUS_DATA_ERROR;
        TraceFailMsg(TLE, T_EXIT, "DeviceGetContext", status);
        WdfRequestComplete(request, status);
        return;
    }

    switch (IoControlCode)
    {
    case GNA_IOCTL_MEM_UNMAP:
        ScoreDeferredUnmap(devCtx, request);
        break;

    default:
        status = STATUS_INVALID_DEVICE_REQUEST;
        TraceFailMsg(TLE, T_EXIT, "(Unknown IOCTL)", status);
        WdfRequestComplete(request, status);
        break;
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);

    return;
}


/******************************************************************************
 * Private Methods
 ******************************************************************************/

static VOID
IoctlMemMap(
    _In_    WDFDEVICE   dev,
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request)
{
    NTSTATUS    status      = STATUS_SUCCESS;
    PAPP_CTX    appCtx      = NULL;
    size_t      inLength    = 0;
    PVOID       inputData   = NULL;
    
    TraceEntry(TLI, T_ENT);

    ASSERT(NULL != devCtx);

    // prevent from double-mapping, verify if mem is not mapped already
    appCtx = GetFileContext(WdfRequestGetFileObject(request));

    // retrieve and store input params
    status = WdfRequestRetrieveOutputBuffer(request, XNN_LYR_DSC_SIZE, &inputData, &inLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveOutputBuffer", status);
        goto ioctl_mm_error;
    }

    PIRP irp = WdfRequestWdmGetIrp(request);
    status = MemoryMap(dev, devCtx, appCtx, irp->MdlAddress, request, (UINT32)inLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "MemoryMap failed", status);
        goto ioctl_mm_error;
    }

    return;

ioctl_mm_error:
    WdfRequestComplete(request, status);
}

static VOID
IoctlMemUnmap(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  unmapReq)
{
    NTSTATUS status = STATUS_SUCCESS;

    TraceEntry(TLI, T_ENT);
    // forward unmap req. to queue
    status = WdfRequestForwardToIoQueue(unmapReq, devCtx->queue);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestForwardToIoQueue", status);
        // complete unmap request in default queue
        WdfRequestComplete(unmapReq, status);
    }
}

static VOID
IoctlNotify(
    _Inout_ WDFREQUEST  request)
{
    TraceEntry(TLI, T_ENT);

    PAPP_CTX appCtx = NULL;

    appCtx = GetFileContext(WdfRequestGetFileObject(request));
    appCtx->notifyRequest = request;

    Trace(TLI, T_EXIT, "Notify request stored in application context");

    return;
}

static VOID
IoctlGetCapabilities(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request)
{
    GNA_CPBLTS * outbuf;
    size_t outbuf_len;

    TraceEntry(TLI, T_ENT);

    NTSTATUS status = STATUS_SUCCESS;

    status = WdfRequestRetrieveOutputBuffer(request, sizeof(GNA_CPBLTS), &outbuf, &outbuf_len);
    if (!NT_SUCCESS(status))
    {

        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveOutputBuffer", status);
        // complete unmap request in default queue
        WdfRequestComplete(request, status);
        return;
    }

    *outbuf = devCtx->cfg.cpblts;

    WdfRequestCompleteWithInformation(request, STATUS_SUCCESS, outbuf_len);
}

#ifdef DRV_DEBUG_INTERFACE
static VOID
IoctlReadReg(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request)
{
    NTSTATUS status       = STATUS_SUCCESS;
    size_t   inputLength  = 0;
    size_t   outputLength = 0;
    PGNA_READREG_IN inputData;
    PGNA_READREG_OUT outputData;

    TraceEntry(TLV, T_ENT);

    status = WdfRequestRetrieveInputBuffer(request, sizeof(PGNA_READREG_IN), &inputData, &inputLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        goto ioctl_read_error;
    }

    status = WdfRequestRetrieveOutputBuffer(request, sizeof(GNA_READREG_OUT), &outputData, &outputLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveOutputBuffer", status);
        goto ioctl_read_error;
    }

    if (0 != inputData->mbarIndex || 
        inputData->regOffset >= devCtx->hw.regsLength ||
        inputData->regOffset % 4 != 0)
    {
        status = STATUS_INVALID_PARAMETER;
        TraceFailMsg(TLE, T_EXIT, "(Invalid read parameters)", status);
        goto ioctl_read_error;
    }
    
    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    outputData->regValue = HwReadReg(devCtx->hw.regs, inputData->regOffset);

    WdfRequestCompleteWithInformation(request, status, outputLength);
    return;

ioctl_read_error:
    WdfRequestComplete(request, status);
}

static VOID
IoctlWriteReg(
    _Inout_ PDEV_CTX    devCtx,
    _Inout_ WDFREQUEST  request)
{
    NTSTATUS    status = STATUS_SUCCESS;
    size_t      inputLength;
    PGNA_WRITEREG_IN inputData;

    TraceEntry(TLV, T_ENT);

    status = WdfRequestRetrieveInputBuffer(request, sizeof(PGNA_WRITEREG_IN), &inputData, &inputLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        goto ioctl_write_error;
    }

    if (0 != inputData->mbarIndex || 
        inputData->regOffset >= devCtx->hw.regsLength ||
        inputData->regOffset % 4 != 0)
    {
        status = STATUS_INVALID_PARAMETER;
        TraceFailMsg(TLE, T_EXIT, "(Invalid write parameters)", status);
        goto ioctl_write_error;
    }

    EventWriteHwRegisterWrite(NULL, __FUNCTION__);
    HwWriteReg(devCtx->hw.regs, inputData->regOffset, inputData->regValue);    

ioctl_write_error:
    WdfRequestComplete(request, status);
}

static VOID
IoctlReadPageDir(
    _Inout_ WDFREQUEST  request)
{
    NTSTATUS status = STATUS_SUCCESS;
    PUINT64 modelId = NULL;
    size_t midLength = 0;
    PGNA_PGDIR_OUT outData = NULL;
    size_t outLength = 0;
    PAPP_CTX appCtx = NULL;
    PMODEL_CTX modelCtx = NULL;

    status = WdfRequestRetrieveInputBuffer(request, sizeof(UINT64), &modelId, &midLength);

    Trace(TLV, T_MEM, "Model id sent from userland: %llu", *modelId);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveInputBuffer", status);
        goto ioctl_readpgdir_error;
    }

    // not compatible data sent from userland
    if (sizeof(*modelId) != midLength)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Bad data sent", status);
        goto ioctl_readpgdir_error;
    }

    // bad model id
    if (*modelId >= APP_MODELS_LIMIT)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Bad model id", status);
        goto ioctl_readpgdir_error;
    }

    status = WdfRequestRetrieveOutputBuffer(request, sizeof(GNA_PGDIR_OUT), &outData, &outLength);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfRequestRetrieveOutputBuffer", status);
        goto ioctl_readpgdir_error;
    }

    appCtx = GetFileContext(WdfRequestGetFileObject(request));
    modelCtx = appCtx->models[*modelId];
    if (NULL == modelCtx)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Model context for given model id does not exist", status);
        goto ioctl_readpgdir_error;
    }

    outData->ptCount = modelCtx->pageTableCount;
    {
        UINT64 i = 0;
        UINT64 copied = 0;
        UINT64 toWrite = 0;

        for (i = 0; i < modelCtx->pageTableCount; ++i)
        {
            outData->l1PhysAddr[i] = modelCtx->ptDir[i].commBuffLa.QuadPart;
            toWrite = (UINT64)(modelCtx->userMemorySize - copied > PAGE_SIZE) ? PAGE_SIZE : modelCtx->userMemorySize - copied;
            if (toWrite > 0)
            {
                memcpy_s(outData->l2PhysAddr + PT_ENTRY_NO * i, PAGE_SIZE, modelCtx->ptDir[i].commBuffVa, PAGE_SIZE);
                copied += toWrite;
            }
        }
    }

    WdfRequestCompleteWithInformation(request, status, outLength);
    return;

ioctl_readpgdir_error:
    WdfRequestComplete(request, status);
}

#endif // DRV_DEBUG_INTERFACE
