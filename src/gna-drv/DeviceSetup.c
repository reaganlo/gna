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

#include "DeviceSetup.h"
#include "devicesetup.tmh"
#include "Hw.h"
#include "Device.h"
#include "IoControl.h"
#include "ScoreProcessor.h"
#include "gna-etw-manifest.h"

#include "wdmguid.h"

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/

/**
 * WDF Event handlers
 */
EVT_WDF_DEVICE_PREPARE_HARDWARE
DevicePrepareHardwareEvnt;

/**
 * WDF Event handlers
 */
EVT_WDF_DEVICE_RELEASE_HARDWARE
DeviceReleaseHardwareEvnt;

/** Device Deletion- cleanup */
EVT_WDF_DEVICE_CONTEXT_CLEANUP
DeviceCleanupCallback;

/**
 * Initializes driver spinlocks
 *
 * @dev                 device handle
 * @devCtx              device context
 */
NTSTATUS
SpinlockInit(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx);

/**
 * Initializes driver request queue
 *
 * @dev                 device handle
 * @devCtx              device context
 */
NTSTATUS
QueueInit(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx);

/**
 * Initializes device interrupts
 *
 * @dev                 device object handle
 * @interrupt           interrupt to initialize
 */
NTSTATUS
InterruptInit(
    _In_    WDFDEVICE   dev,
    _Inout_ WDFINTERRUPT* interrupt);

/**
 * Initializes request timeout timer
 *
 * @dev                 device handle
 */
NTSTATUS
TimerInit(
    _In_     WDFDEVICE  dev);

/**
 * Initializes device DMA enabler for DMA data transfers
 *
 * @dev                 device object handle
 * @devCtx              device context
 */
NTSTATUS
DeviceDmaInit(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx);

#ifdef ALLOC_PRAGMA
#pragma alloc_text (PAGE, DeviceInit)
#pragma alloc_text (PAGE, SpinlockInit)
#pragma alloc_text (PAGE, DevicePrepareHardwareEvnt)
#pragma alloc_text (PAGE, DeviceReleaseHardwareEvnt)
#pragma alloc_text (PAGE, DeviceDmaInit)
#pragma alloc_text (PAGE, QueueInit)
#pragma alloc_text (PAGE, InterruptInit)
#pragma alloc_text (PAGE, TimerInit)
#pragma alloc_text (PAGE, GetDeviceCapabilities)
#endif

/******************************************************************************
 * Public Methods
 ******************************************************************************/

NTSTATUS
DeviceInit(
    _Inout_ PWDFDEVICE_INIT     devInit)
{
    WDF_PNPPOWER_EVENT_CALLBACKS    pnpPowerCallbacks;
    WDF_OBJECT_ATTRIBUTES           deviceAttributes;
    WDF_OBJECT_ATTRIBUTES           fileAttributes;
    WDF_FILEOBJECT_CONFIG           fileConfig;
    WDF_DEVICE_POWER_POLICY_IDLE_SETTINGS idleSettings;
    PDEV_CTX    devCtx;
    WDFDEVICE   dev;
    NTSTATUS    status;

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);

    WdfDeviceInitSetIoType(devInit, WdfDeviceIoDirect);

    WDF_PNPPOWER_EVENT_CALLBACKS_INIT(&pnpPowerCallbacks);
    pnpPowerCallbacks.EvtDevicePrepareHardware = DevicePrepareHardwareEvnt;
    pnpPowerCallbacks.EvtDeviceReleaseHardware = DeviceReleaseHardwareEvnt;
    pnpPowerCallbacks.EvtDeviceD0Entry = DeviceD0EntryEvnt;
    pnpPowerCallbacks.EvtDeviceD0Exit = DeviceD0ExitEvnt;
    WdfDeviceInitSetPnpPowerEventCallbacks(devInit, &pnpPowerCallbacks);

    WDF_FILEOBJECT_CONFIG_INIT(&fileConfig,
        FileCreateEvnt,
        FileCloseEvnt,
        FileCleanupEvnt);

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&fileAttributes, APP_CTX);
    WdfDeviceInitSetFileObjectConfig(devInit,
        &fileConfig,
        &fileAttributes);

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&deviceAttributes, DEV_CTX);
    deviceAttributes.EvtCleanupCallback = DeviceCleanupCallback;

    status = WdfDeviceCreate(&devInit, &deviceAttributes, &dev);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDeviceCreate", status);
        goto cleanup;
    }

    devCtx = DeviceGetContext(dev);

    // initialize states
    RtlZeroMemory(&devCtx->app, sizeof(APP_STATE));
    RtlZeroMemory(&devCtx->req, sizeof(REQ_STATE));
    devCtx->cfg.d0i3Enabled = FALSE;

    status = TimerInit(dev);
    if (!NT_SUCCESS(status)) goto cleanup;

    status = InterruptInit(dev, &devCtx->cfg.interrupt);
    if (!NT_SUCCESS(status)) goto cleanup;

    status = DeviceDmaInit(dev, devCtx);
    if (!NT_SUCCESS(status)) goto cleanup;

    status = SpinlockInit(dev, devCtx);
    if (!NT_SUCCESS(status)) goto cleanup;

    //
    // Create a device interface so that applications can find and talk
    // to us.
    //
    status = WdfDeviceCreateDeviceInterface(
        dev,
        &GUID_DEVINTERFACE_GNA_DRV,
        NULL // ReferenceString
        );

    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDeviceCreateDeviceInterface", status);
        goto cleanup;
    }

    //
    // Initialize the I/O Package and any Queues
    //
    status = QueueInit(dev, devCtx);
    if (!NT_SUCCESS(status)) goto cleanup;

    // Set power idling policy
    WDF_DEVICE_POWER_POLICY_IDLE_SETTINGS_INIT(&idleSettings, IdleCannotWakeFromS0);

    idleSettings.ExcludeD3Cold = WdfFalse;
    idleSettings.Enabled = WdfTrue;
    idleSettings.PowerUpIdleDeviceOnSystemWake = WdfTrue;

    status = WdfDeviceAssignS0IdleSettings(dev, &idleSettings);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDeviceAssignS0IdleSettings", status);
        goto cleanup;
    }

    EventWriteDeviceInitSuccess(NULL);
    return status;

cleanup:
    TraceReturn(TLE, T_EXIT, status);
    EventWriteDeviceInitFail(NULL, status);
    return status;
}

/******************************************************************************
 * Private Methods
 ******************************************************************************/

GNA_CPBLTS
GetDeviceCapabilities(
    _In_ WDFDEVICE dev,
    _In_ PDEV_CTX  devCtx)
{
    UNREFERENCED_PARAMETER(dev);
    UNREFERENCED_PARAMETER(devCtx);
    PAGED_CODE();

    /*WDF_OBJECT_ATTRIBUTES attributes;
    WDFMEMORY memory;
    size_t hwid_memory;
    PVOID hwid_buf;*/
    GNA_CPBLTS cpblts;
    RtlZeroMemory(&cpblts, sizeof(GNA_CPBLTS));

    cpblts.hwInBuffSize = HwReadInBuffSize(devCtx->hw.regs);
    // NOTE: temporary workaround for missing HWID
        // TODO: remove when HWID is assigned
    cpblts.device_type = GNA_TIGERLAKE;
    Trace(TLI, T_MEM, "TGL found");

    /*WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
    attributes.ParentObject = dev;
    NTSTATUS status = WdfDeviceAllocAndQueryProperty(dev, DevicePropertyHardwareID, NonPagedPoolNx, &attributes, &memory);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDeviceAllocAndQueryProperty", status);
    }
    else
    {
        hwid_buf = WdfMemoryGetBuffer(memory, &hwid_memory);
        if (hwid_buf == NULL) {
            TraceFailMsg(TLE, T_EXIT, "WdfMemoryGetBuffer returned NULL", status);
        }
        else
        {
            UNICODE_STRING hwid, hwid2;
            RtlInitUnicodeString(&hwid, hwid_buf);
            Trace(TLI, T_MEM, "Device name length: %Iu", hwid_memory);
            Trace(TLI, T_MEM, "Device name (unicode): %wZ", &hwid);

            RtlInitUnicodeString(&hwid2, L"PCI\\VEN_8086&DEV_3190");
            Trace(TLV, T_MEM, "GLK unicode: %wZ", &hwid2);
            LONG cmp_res = RtlCompareUnicodeString(&hwid2, &hwid, TRUE);
            if (cmp_res > -23)
            {
                RtlInitUnicodeString(&hwid2, L"PCI\\VEN_8086&DEV_5A11");
                Trace(TLV, T_MEM, "CNL unicode: %wZ", &hwid2);
                cmp_res = RtlCompareUnicodeString(&hwid2, &hwid, TRUE);

                if (cmp_res > -23)
                {
                    Trace(TLI, T_MEM, "Unknown device found");
                    cpblts.device_type = GNA_NUM_DEVICE_TYPES;
                }
                else
                {
                    Trace(TLI, T_MEM, "CNL found");
                    cpblts.device_type = GNA_CANNONLAKE;
                }
            }
            else
            {
                Trace(TLI, T_MEM, "GLK found");
                cpblts.device_type = GNA_GEMINILAKE;
            }
        }
    }*/
    return cpblts;
}

NTSTATUS
DevicePrepareHardwareEvnt(
    WDFDEVICE           dev,
    WDFCMRESLIST        resources,
    WDFCMRESLIST        resTranslated)
{
    NTSTATUS status = STATUS_SUCCESS;
    PDEV_CTX devCtx;
    ULONG    i;
    PCM_PARTIAL_RESOURCE_DESCRIPTOR desc;
    BOOLEAN  hwRegsFound = FALSE;

    UNREFERENCED_PARAMETER(resources);

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);
    EventWriteGenericFunctionEntry(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(dev);

    //Get BusInterface to PCI-CFG space
    status = WdfFdoQueryForInterface( dev, &GUID_BUS_INTERFACE_STANDARD,
         (PINTERFACE)&devCtx->hw.busInterface, sizeof(BUS_INTERFACE_STANDARD), 1, NULL );

     if(!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfFdoQueryForInterface: Getting BusInterface", status);
        EventWriteDevicePrepareHWfail(NULL, "WdfFdoQueryForInterface: Getting BusInterface", status);
        return STATUS_DEVICE_CONFIGURATION_ERROR;
    }

    for (i = 0; i < WdfCmResourceListGetCount(resTranslated); ++i)
    {
        desc = WdfCmResourceListGetDescriptor(resTranslated, i);
        if (NULL == desc)
        {
            TraceFailMsg(TLE, T_EXIT, "WdfResourceListGetDescriptor", STATUS_DEVICE_CONFIGURATION_ERROR);
            EventWriteDevicePrepareHWfail(NULL, "WdfResourceListGetDescriptor", STATUS_DEVICE_CONFIGURATION_ERROR);
            return STATUS_DEVICE_CONFIGURATION_ERROR;
        }
        switch (desc->Type)
        {
        case CmResourceTypeMemory:
            Trace(TLI, T_MEM, "%!FUNC! Memory Resource @ %x:%x length %d",
                desc->u.Memory.Start.HighPart, desc->u.Memory.Start.LowPart,
                desc->u.Memory.Length);
            if (!hwRegsFound)
            {
#if (NTDDI_VERSION >= NTDDI_WINTHRESHOLD)
                devCtx->hw.regs = (P_HW_REGS)MmMapIoSpaceEx(desc->u.Memory.Start, desc->u.Memory.Length, PAGE_NOCACHE | PAGE_READWRITE);
#else
                // TODO resolve this once the project is fully moved to VS2015
                #pragma warning(suppress: 30029)
                devCtx->hw.regs = (P_HW_REGS)MmMapIoSpace(desc->u.Memory.Start, desc->u.Memory.Length, MmNonCached);
#endif
                devCtx->hw.regsLength = desc->u.Memory.Length;
                hwRegsFound = TRUE;
                Trace(TLV, T_MEM, "%!FUNC! Memory assigned as HW registers Mapped @ %p",
                    devCtx->hw.regs);
            }
            break;
        default:
            Trace(TLW, T_MEM, "%!FUNC! Unknown Resource type %d", desc->Type);
            break;
        }
    }
    if (4096 != devCtx->hw.regsLength || devCtx->hw.regsLength != sizeof(HW_REGS))
    {
        Trace(TLE, T_MEM, "%!FUNC! Invalid HW Regs size: @ %u", sizeof(HW_REGS));
        EventWriteDevicePrepareHWfail(NULL, "Invalid HW Regs size", STATUS_DEVICE_CONFIGURATION_ERROR);
        return STATUS_DEVICE_CONFIGURATION_ERROR;
    }

    // discover driver capabilities
    devCtx->cfg.cpblts = GetDeviceCapabilities(dev, devCtx);


    TraceReturn(TLE, T_EXIT, status);
    return status;
}

NTSTATUS
DeviceReleaseHardwareEvnt(
    WDFDEVICE           dev,
    WDFCMRESLIST        resTranslated)
{
    NTSTATUS status = STATUS_SUCCESS;
    PDEV_CTX devCtx;

    UNREFERENCED_PARAMETER(resTranslated);

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);
    EventWriteGenericFunctionEntry(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(dev);
    // release HW config
    if (devCtx->hw.regs)
    {
        MmUnmapIoSpace(devCtx->hw.regs, devCtx->hw.regsLength);
    }
    RtlZeroMemory(&devCtx->hw, sizeof(DEV_HW));

    TraceReturn(TLE, T_EXIT, status);
    return status;
}

VOID
DeviceCleanupCallback(
    _In_    WDFOBJECT   dev)
{
    PDEV_CTX devCtx;

    TraceEntry(TLI, T_ENT);
    EventWriteGenericFunctionEntry(NULL, __FUNCTION__);

    devCtx = DeviceGetContext(dev);
}


NTSTATUS
DeviceDmaInit(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx)
{
    NTSTATUS status = STATUS_SUCCESS;
    WDF_DMA_ENABLER_CONFIG dmaConfig;

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);

    WdfDeviceSetAlignmentRequirement(dev, PAGE_SIZE-1);
    WDF_DMA_ENABLER_CONFIG_INIT(&dmaConfig,
        WdfDmaProfileScatterGather64,
        HW_MAX_MEM_SIZE);
    status = WdfDmaEnablerCreate(dev,
        &dmaConfig,
        WDF_NO_OBJECT_ATTRIBUTES,
        &devCtx->cfg.dmaEnabler);

    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDmaEnablerCreate", status);
        EventWriteDmaEnablerCreateFailed(NULL, status);
    }
    else
    {
        Trace(TLI, T_INIT, "%!FUNC! DmaEnabler created 0x%p maxXferSize %lld", devCtx->cfg.dmaEnabler, dmaConfig.MaximumLength);
        EventWriteDmaEnablerCreated(NULL);
    }

    return status;
}

NTSTATUS
SpinlockInit(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx)
{
    NTSTATUS              status;
    WDF_OBJECT_ATTRIBUTES attributes;

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);

    // initialize spinlocks
    WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
    attributes.ParentObject = dev;
    status = WdfSpinLockCreate(&attributes, &devCtx->app.appLock);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfSpinLockCreate(appLock)", status);
        EventWriteSpinLockCreateFailed(NULL, "appLock", status);
        return status;
    }
    else
    {
        EventWriteSpinLockCreated(NULL, "appLock");
    }

    status = WdfSpinLockCreate(&attributes, &devCtx->req.reqLock);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfSpinLockCreate(reqLock)", status);
        EventWriteSpinLockCreateFailed(NULL, "reqLock", status);
        return status;
    }
    else
    {
        EventWriteSpinLockCreated(NULL, "reqLock");
    }

    status = WdfSpinLockCreate(&attributes, &devCtx->cfg.pwrLock);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfSpinLockCreate(pwrLock)", status);
        EventWriteSpinLockCreateFailed(NULL, "pwrLock", status);
        return status;
    }
    else
    {
        EventWriteSpinLockCreated(NULL, "pwrLock");
    }

    return status;
}

NTSTATUS
QueueInit(
    _In_    WDFDEVICE   dev,
    _In_    PDEV_CTX    devCtx)
{
    WDFQUEUE            queue;
    NTSTATUS            status;
    WDF_IO_QUEUE_CONFIG queueConfig;
    WDF_OBJECT_ATTRIBUTES attributes;

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);

    //
    // Configure a default queue so that requests that are not
    // configure-forwarded using WdfDeviceConfigureRequestDispatching to goto
    // other queues get dispatched here.
    //
    WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&queueConfig, WdfIoQueueDispatchParallel);
    queueConfig.EvtIoDeviceControl = IoctlDispatcher;
    queueConfig.PowerManaged = WdfTrue;

    status = WdfIoQueueCreate(
                 dev,
                 &queueConfig,
                 WDF_NO_OBJECT_ATTRIBUTES,
                 &queue
                 );

    if( !NT_SUCCESS(status) )
    {
        TraceFailMsg(TLE, T_EXIT, "WdfIoQueueCreate(default)", status);
        EventWriteQueueMainCreateFailed(NULL, status);
        return status;
    }
    EventWriteQueueMainCreated(NULL);

    WDF_IO_QUEUE_CONFIG_INIT(&queueConfig, WdfIoQueueDispatchSequential);
    queueConfig.PowerManaged = WdfTrue;
    queueConfig.EvtIoDeviceControl = IoctlDeferred;
    queueConfig.EvtIoWrite = ScoreSubmitEvnt;
    // TODO: add EvtIoStop callback to cancel current requests on d3
    //queueConfig.EvtIoStop

    WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
    attributes.ParentObject = dev;
    attributes.SynchronizationScope = WdfSynchronizationScopeQueue;

    status = WdfIoQueueCreate(
                dev,
                &queueConfig,
                &attributes,
                &devCtx->queue);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfIoQueueCreate(ioctlPendingQueue)", status);
        EventWriteQueueAuxCreateFailed(NULL, status);
        return status;
    }

    status = WdfDeviceConfigureRequestDispatching(
                dev,
                devCtx->queue,
                WdfRequestTypeWrite);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDeviceConfigureRequestDispatching", status);
        EventWriteQueueAuxCreateFailed(NULL, status);
        return status;
    }

    EventWriteQueueAuxCreated(NULL);
    return status;
}

NTSTATUS
InterruptInit(
    _In_    WDFDEVICE       dev,
    _Inout_ WDFINTERRUPT*   interrupt)
{
    NTSTATUS status = STATUS_SUCCESS;
    WDF_INTERRUPT_CONFIG intCfg;

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);

    WDF_INTERRUPT_CONFIG_INIT(&intCfg,
                              InterruptIsrEvnt,
                              InterruptDpcEvnt);
    intCfg.EvtInterruptEnable = WDF_NO_EVENT_CALLBACK;
    intCfg.EvtInterruptDisable = WDF_NO_EVENT_CALLBACK;

    status = WdfInterruptCreate(dev,
                                &intCfg,
                                WDF_NO_OBJECT_ATTRIBUTES,
                                interrupt);

    if (NT_SUCCESS(status))
        EventWriteInterruptCreated(NULL);
    else
        EventWriteInterruptCreateFailed(NULL, status);

    TraceReturn(TLE, T_EXIT, status);
    return status;
}

NTSTATUS
TimerInit(
    _In_    WDFDEVICE   dev)
{
    WDF_TIMER_CONFIG        timerConfig;
    NTSTATUS                status;
    WDF_OBJECT_ATTRIBUTES   timerAttributes;
    PDEV_CTX                devCtx;

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);
    WDF_TIMER_CONFIG_INIT(&timerConfig, ScoreTimeoutEvnt);
    WDF_OBJECT_ATTRIBUTES_INIT(&timerAttributes);
    timerAttributes.ParentObject = dev;

    devCtx = DeviceGetContext(dev);
    status = WdfTimerCreate(&timerConfig, &timerAttributes, &devCtx->timeout);
    return status;
}