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

    driver.c

Abstract:

    This file contains the driver entry points and callbacks.

Environment:

    Kernel-mode Driver Framework

--*/

#include "Driver.h"
#include "driver.tmh"
#include "DeviceSetup.h"
#include "gna-etw-manifest.h"

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/

/**
 * WDF DRIVER initialize event
 */
DRIVER_INITIALIZE DriverEntry;

/**
 * WDF DRIVER device add event
 */
EVT_WDF_DRIVER_DEVICE_ADD
DriverDeviceAddEvnt;

/**
 * WDF DRIVER cleanup event
 */
EVT_WDF_OBJECT_CONTEXT_CLEANUP
DriverContextCleanupEvnt;

#ifdef ALLOC_PRAGMA
#pragma alloc_text (INIT, DriverEntry)
#pragma alloc_text (PAGE, DriverDeviceAddEvnt)
#pragma alloc_text (PAGE, DriverContextCleanupEvnt)
#endif

/******************************************************************************
 * Public Methods
 ******************************************************************************/

NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT      drvObj,
	_In_ PUNICODE_STRING     regPath
    )
/*++

Routine Description:
    DriverEntry initializes the driver and is the first routine called by the
    system after the driver is loaded. DriverEntry specifies the other entry
    points in the function driver, such as EvtDevice and DriverUnload.

Parameters Description:

    drvObj - represents the instance of the function driver that is loaded
    into memory. DriverEntry must initialize members of DriverObject before it
    returns to the caller. DriverObject is allocated by the system before the
    driver is loaded, and it is released by the system after the system unloads
    the function driver from memory.

    regPath - represents the driver specific path in the Registry.
    The function driver can use the path to store driver related data between
    reboots. The path does not store hardware instance specific data.

Return Value:

    STATUS_SUCCESS if successful,
    STATUS_UNSUCCESSFUL otherwise.

--*/
{
    WDF_DRIVER_CONFIG       config;
    NTSTATUS                status;
    WDF_OBJECT_ATTRIBUTES   drvAttributes;
    WDFDRIVER               drv;

    //
    // Initialize WPP Tracing
    //
    WPP_INIT_TRACING( drvObj, regPath );

    TraceEntry(TLI, T_ENT);

    EventRegisterIntel_GNA_Driver();
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    //
    // Register a cleanup callback so that we can call WPP_CLEANUP when
    // the framework driver object is deleted during driver unload.
    WDF_OBJECT_ATTRIBUTES_INIT(&drvAttributes);
    drvAttributes.EvtCleanupCallback = DriverContextCleanupEvnt;

    WDF_DRIVER_CONFIG_INIT(&config,
                           DriverDeviceAddEvnt
                           );

    status = WdfDriverCreate(drvObj,
                             regPath,
                             &drvAttributes,
                             &config,
                             &drv
                             );

    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfDriverCreate", status);
        WPP_CLEANUP(drvObj);

        EventWriteDriverCreateFail(NULL, status);
        EventUnregisterIntel_GNA_Driver();
    }
    else
    {
        EventWriteDriverEntrySuccess(NULL);
    }

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
    return status;
}

/******************************************************************************
 * Private Methods
 ******************************************************************************/

NTSTATUS
DriverDeviceAddEvnt(
    WDFDRIVER           drv,
    PWDFDEVICE_INIT     devInit
    )
/*++
Routine Description:

    EvtDeviceAdd is called by the framework in response to AddDevice
    call from the PnP manager. We create and initialize a device object to
    represent a new instance of the device.

Arguments:

    drv - Handle to a framework driver object created in DriverEntry

    devInit - Pointer to a framework-allocated WDFDEVICE_INIT structure.

Return Value:

    NTSTATUS

--*/
{
    NTSTATUS status;

    UNREFERENCED_PARAMETER(drv);

    PAGED_CODE();

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    status = DeviceInit(devInit);

    TraceReturn(TLE, T_EXIT, status);
    EventWriteDriverApiEnd(NULL, __FUNCTION__);
    return status;
}

VOID
DriverContextCleanupEvnt(
    WDFOBJECT           drvObj
    )
/*++
Routine Description:

    Free all the resources allocated in DriverEntry.

Arguments:

    drvObj - handle to a WDF Driver object.

Return Value:

    VOID.

--*/
{
    /**
     * Rationale for warning suppression:
     * EvtCleanupCallback of WDFDRIVER object is called at IRQL = PASSIVE_LEVEL
     */
    #pragma warning(suppress: 28118)
    PAGED_CODE ();

    TraceEntry(TLI, T_ENT);
    EventWriteDriverApiBegin(NULL, __FUNCTION__);

    //
    // Stop WPP Tracing
    //
    WPP_CLEANUP( WdfDriverWdmGetDriverObject(drvObj) );

    EventWriteDriverApiEnd(NULL, __FUNCTION__);
    EventUnregisterIntel_GNA_Driver();
}
