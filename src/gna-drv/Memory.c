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

#include "Memory.h"
#include "Memory.tmh"
#include "Hw.h"
#include "gna-etw-manifest.h"

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/
NTSTATUS
ModelDescInit(
    _In_ PDEV_CTX     devCtx,
    _In_ PMODEL_CTX   modelCtx);
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

/******************************************************************************
 * Public Methods
 ******************************************************************************/

NTSTATUS
MemoryMap(
    _In_    WDFDEVICE    dev,
    _In_    PDEV_CTX     devCtx,
    _In_    PAPP_CTX     appCtx,
    _In_    PMDL         pMdl,
    _In_    WDFREQUEST   mapRequest,
    _In_    UINT32       length)
{
    NTSTATUS    status = STATUS_SUCCESS;
    PMODEL_CTX  modelCtx = NULL;
    PVOID       usrBuffer = NULL;
    PVOID       dmaVA = NULL;
    PDMA_ADAPTER pDmaAdapter = NULL;
    NPAGED_LOOKASIDE_LIST lsList;
    BOOLEAN     lsListInitDone = FALSE;
    BOOLEAN     SglBuildDone = FALSE;
    PVOID       pSglBuffer = NULL;
    KIRQL       Irql;
    ULONG       SglSize;
    ULONG       sglMapRegs;
    status_t    sts = GNA_SUCCESS; // status of internal calls
    UINT32      nPTables = 0;// number of required page tables
    UINT32      nPTentries = 0;// number of required page tables entries

    TraceEntry(TLI, T_ENT);

    dmaVA = MmGetMdlVirtualAddress(pMdl);
    usrBuffer = MmGetSystemAddressForMdlSafe(pMdl, NormalPagePriority | MdlMappingNoExecute);
    Trace(TLI, T_MEM, "User buffer address: %p", usrBuffer);
    if (NULL == usrBuffer)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "MmGetSystemAddressForMdlSafe returned NULL", status);
        goto mem_map_error;
    }

    UINT64 modelId = *(UINT64*)usrBuffer;
    Trace(TLI, T_MEM, "Model id saved in user buffer: %lld", modelId);

    // bad model id
    if (modelId >= APP_MODELS_LIMIT)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Bad model id", status);
        goto mem_map_error;
    }

    modelCtx = appCtx->models[modelId];
    if (NULL == modelCtx)
    {
        modelCtx = (PMODEL_CTX) ExAllocatePoolWithTag(NonPagedPoolNx, sizeof(MODEL_CTX), MEM_POOL_TAG);

        modelCtx->pMdl = pMdl;
        modelCtx->mmapRequest = mapRequest;
        modelCtx->userMemoryBaseVA = usrBuffer;
        modelCtx->userMemorySize = length;
        modelCtx->requestConfigId = -1;
        appCtx->models[modelId] = modelCtx;

        Trace(TLI, T_MEM, "Memory mapping model with modelId = %lld", modelId);
    }
    // model already mapped
    else
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "Model with provided model id already exists", status);
        goto mem_map_error;
    }

    sts = CheckMapConfigParameters(dmaVA, length);
    if (GNA_SUCCESS != sts)
    {
        Trace(TLE, T_EXIT, "%!FUNC!: CheckMapConfigParameters failed with %d", sts);
        EventWriteMemoryMapFail(NULL, status);
        goto mem_map_error;
    }

    pDmaAdapter = WdfDmaEnablerWdmGetDmaAdapter(devCtx->cfg.dmaEnabler, WdfDmaDirectionReadFromDevice);
    if (NULL == pDmaAdapter)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "WdfDmaEnablerWdmGetDmaAdapter", status);
        goto mem_map_error;
    }

    status = pDmaAdapter->DmaOperations->CalculateScatterGatherList(pDmaAdapter,
        modelCtx->pMdl,
        dmaVA,
        length,
        &SglSize,
        &sglMapRegs);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "CalculateScatterGatherList", status);
        goto mem_map_error;
    }

    ExInitializeNPagedLookasideList(&lsList, NULL, NULL, POOL_NX_ALLOCATION, SglSize, MEM_POOL_TAG, 0);
    lsListInitDone = TRUE;
    pSglBuffer = ExAllocateFromNPagedLookasideList(&lsList);
    if (NULL == pSglBuffer)
    {
        status = STATUS_UNSUCCESSFUL;
        TraceFailMsg(TLE, T_EXIT, "ExAllocateFromNPagedLookasideList", status);
        goto mem_map_error;
    }

    KeRaiseIrql(DISPATCH_LEVEL, &Irql);

/**
 * Rationale for warning suppression:
 * Parameter 7 can be null as callback function ProcessSGList ignores argument context
 */
#pragma warning(suppress: 6387)
        status = pDmaAdapter->DmaOperations->BuildScatterGatherList(pDmaAdapter,
            WdfDeviceWdmGetDeviceObject(dev),
            modelCtx->pMdl,
            dmaVA,
            length,
            ProcessSGList,
            NULL,
            FALSE,
            pSglBuffer,
            SglSize);
    KeLowerIrql(Irql);

    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "BuildScatterGatherList", status);
        goto mem_map_error;
    }
    SglBuildDone = TRUE;

    // padd number of entries to 64B and add additional 32entries (128B)
    // for prefetching mechanism in HW to work properly
    nPTentries = 32 +
        (((int)(((DIV_CEIL(length, PAGE_SIZE)) - 1) / 16) + 1) * 16);
    nPTables = DIV_CEIL(nPTentries, PT_ENTRY_NO);
    Trace(TLI, T_MEM, "%!FUNC!: Page Table Entries Count %u", nPTentries);
    Trace(TLI, T_MEM, "%!FUNC!: Page Table Count %u", nPTables);
    if (nPTables > PT_DIR_SIZE + 1)
    {
        TraceFailMsg(TLE, T_EXIT, "Number of PageTable entries and Page Tables too large!", status);
        goto mem_map_error;
    }

    //allocate L1 area
    for (modelCtx->pageTableCount = 0; modelCtx->pageTableCount < nPTables; ++modelCtx->pageTableCount)
    {
        P_PT_DIR pageTable = &modelCtx->ptDir[modelCtx->pageTableCount];
        status = WdfCommonBufferCreate(devCtx->cfg.dmaEnabler,
            PAGE_SIZE,
            WDF_NO_OBJECT_ATTRIBUTES,
            &pageTable->commBuff);
        if (!NT_SUCCESS(status))
        {
            TraceFailMsg(TLE, T_EXIT, "WdfCommonBufferCreate", status);
            goto mem_map_error;
        }
        pageTable->commBuffVa = WdfCommonBufferGetAlignedVirtualAddress(pageTable->commBuff);
        pageTable->commBuffLa = WdfCommonBufferGetAlignedLogicalAddress(pageTable->commBuff);
        RtlZeroMemory(pageTable->commBuffVa, PAGE_SIZE);
    }
    Trace(TLI, T_MEM, "%!FUNC!: L1 Table pages allocated: %d", modelCtx->pageTableCount);

    //copy L2 addresses to L1 area
    {
        PSCATTER_GATHER_LIST pSgList = (PSCATTER_GATHER_LIST) pSglBuffer;
        PSCATTER_GATHER_ELEMENT pSgListElement = pSgList->Elements;

        P_PT_DIR pageTable = modelCtx->ptDir;
        ULONG32 *pageTableEntry = (ULONG32*) pageTable->commBuffVa;
        ULONG32* pageTableEntriesEnd = pageTableEntry + PT_ENTRY_NO;

        Trace(TLI, T_MEM, "%!FUNC!: SGLs used: %d", pSgList->NumberOfElements);

        while (pSgListElement < pSgList->Elements + pSgList->NumberOfElements)
        {
            // need to maintain chunks of page size, so if there is larger physical area, need to split it
            while (pSgListElement->Length > 0)
            {
                ULONG64 address = (ULONG64) pSgListElement->Address.QuadPart;
                ASSERT(address%PAGE_SIZE == 0);
                address /= PAGE_SIZE;

                Trace(TLV, T_MEM, "%!FUNC!: App page address / 4KB %llX stored @ %p", address, pageTableEntry);
                *pageTableEntry++ = (ULONG32) address;

                // check if need to switch to next page
                if (pageTableEntry == pageTableEntriesEnd)
                {
                    pageTableEntry = (ULONG32*) (++pageTable)->commBuffVa;
                    pageTableEntriesEnd = pageTableEntry + PT_ENTRY_NO;
                }

                if (pSgListElement->Length > PAGE_SIZE)
                {
                    pSgListElement->Length -= PAGE_SIZE;
                    pSgListElement->Address.QuadPart += PAGE_SIZE;
                }
                else
                {
                    pSgListElement->Length = 0;
                }

            }
            ++pSgListElement;
        }
    }

    // prepare and store mmu config in app ctx for later copying into hw descriptor
    // HW configuration of mapping executed on app context switch before scoring start
    ModelDescInit(devCtx, modelCtx);
    HwPrepareMmuConfig(modelCtx);

    Trace(TLI, T_MEM, "%!FUNC! HW Memory mapped successfully");

mem_map_error:
    // This is common mem_map_error:
    KeRaiseIrql(DISPATCH_LEVEL, &Irql);
        if (SglBuildDone)
        {
            pDmaAdapter->DmaOperations->PutScatterGatherList(pDmaAdapter,
                (PSCATTER_GATHER_LIST) pSglBuffer,
                TRUE /*? not sure*/);
            SglBuildDone = FALSE;
        }
    KeLowerIrql(Irql);

    if (NULL != pSglBuffer)
    {
        ExFreeToNPagedLookasideList(&lsList, pSglBuffer);
        pSglBuffer = NULL;
    }
    if (lsListInitDone)
    {
        ExDeleteNPagedLookasideList(&lsList);
        lsListInitDone = FALSE;
    }

    // This is error condition part of mem_map_error:
    if (!NT_SUCCESS(status))
    {
        TraceFail(TLE, T_EXIT, status);
        EventWriteMemoryMapFail(NULL, status);

        if (NULL != modelCtx)
        {
            MemoryMapRelease(appCtx, modelCtx);
        }
    }
    else
    {
        EventWriteMemoryMapSuccess(NULL);
    }

    if (appCtx->notifyRequest != WDF_NO_HANDLE)
    {
        WdfRequestComplete(appCtx->notifyRequest, status);
        appCtx->notifyRequest = WDF_NO_HANDLE;
        Trace(TLI, T_EXIT, "Notify request is completed with status: %d", status);
    }

    return status;
}

VOID
MemoryMapRelease(
    _Inout_ PAPP_CTX              appCtx,
    _Inout_ PMODEL_CTX            modelCtx)
{
    TraceEntry(TLI, T_ENT);

    UINT64 modelId = modelCtx->modelId;
    Trace(TLI, T_MEM, "Memory unmapping model with modelId = %lld", modelId);

    ULONG i;
    for (i = 0; i < modelCtx->pageTableCount; ++i)
    {
        WdfObjectDelete(modelCtx->ptDir[i].commBuff);
    }
    RtlZeroMemory(modelCtx->ptDir, sizeof(modelCtx->ptDir));
    modelCtx->pageTableCount = 0;

    ModelDescRelease(&modelCtx->desc);

    if (WDF_NO_HANDLE != modelCtx->mmapRequest)
    {
        WdfRequestComplete(modelCtx->mmapRequest, STATUS_SUCCESS);
        ExFreePool(modelCtx);
        appCtx->models[modelId] = NULL;
    }

    EventWriteMemoryReleased(NULL);
}

/******************************************************************************
 * Private Methods
 ******************************************************************************/

NTSTATUS
ModelDescInit(
    _In_ PDEV_CTX     devCtx,
    _In_ PMODEL_CTX   modelCtx)
{
    NTSTATUS status     = STATUS_SUCCESS;
    PAGED_CODE();

    // prepare private configuration memory in common buffer
    status = WdfCommonBufferCreate(devCtx->cfg.dmaEnabler,
        PRV_CFG_SIZE,
        WDF_NO_OBJECT_ATTRIBUTES,
        &modelCtx->desc.buffer);
    if (!NT_SUCCESS(status))
    {
        TraceFailMsg(TLE, T_EXIT, "WdfCommonBufferCreate", status);
        return status;
    }

    modelCtx->desc.va = WdfCommonBufferGetAlignedVirtualAddress(modelCtx->desc.buffer);
    modelCtx->desc.la = WdfCommonBufferGetAlignedLogicalAddress(modelCtx->desc.buffer);
    RtlZeroMemory(modelCtx->desc.va, PRV_CFG_SIZE);
    modelCtx->desc.la.QuadPart /= PAGE_SIZE;
    ASSERTMSG("DeviceDescInit Logical Descriptor address > 32bits",
        (LONG64)(modelCtx->desc.la.QuadPart) < MAXUINT32 );

    return status;
}

VOID
ModelDescRelease(
    _In_    PHW_DESC    desc)
{
    if (WDF_NO_HANDLE != desc->buffer)
    {
        WdfObjectDelete(desc->buffer);
        RtlZeroMemory(desc, sizeof(HW_DESC));
    }
}

status_t
CheckMapConfigParameters(
    _In_    PVOID     usrBuffer,
    _In_    UINT32    length)
{
    TraceEntry(TLI, T_ENT);

    ERRCHECKP(0 == length || length > HW_MAX_MEM_SIZE, GNA_INVALIDMEMSIZE);
    ERRCHECKP(NULL == usrBuffer, GNA_NULLARGNOTALLOWED);
    ERRCHECKP(0 != (PtrToInt(Ptr64ToPtr(usrBuffer)) % PAGE_SIZE), GNA_BADMEMALIGN);
    return GNA_SUCCESS;
}

VOID
ProcessSGList(
    _In_    PDEVICE_OBJECT          devObj,
    _In_    PIRP                    irp,
    _In_    PSCATTER_GATHER_LIST    scatterGather,
    _In_    PVOID                   context
    )
{
    UNREFERENCED_PARAMETER(devObj);
    UNREFERENCED_PARAMETER(irp);
    UNREFERENCED_PARAMETER(scatterGather);
    UNREFERENCED_PARAMETER(context);
    return;
}