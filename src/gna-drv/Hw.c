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

#include "Driver.h"
#include "Hw.h"
#include "hw.tmh"
#include "gna-etw-manifest.h"

/**
 * Rationale for warning suppression:
 * The constant argument should instead be variable, however it would slow down read and write operation.
 */
#pragma warning(disable: 28138)

/**
 * Busy-Wait-loop "time" counter
 **/
#define     CMD_CMPLT_TIMEOUT       100

/******************************************************************************
 * Private Methods declaration
 ******************************************************************************/

/**
 * Pads size to given number
 *
 * @size                size to be padded
 * @pad                 number of bytes to pad
 * @return              memory size (int bytes) padded to given value
 */
#define PAD(size, pad)  (((int)((size) + pad -1) / pad) * pad)

/**
 * Gets and traces HW registers for debug purposes
 *
 * @regs                hardware registers MMIO address
 */
#ifdef _DEBUG
VOID    getRegs(_In_ P_HW_REGS regs);
#else
#define getRegs(...)    // unavailable in release config
#endif

#define _READ(reg)          READ_REGISTER_ULONG ((PULONG)&reg)
#define _WRITE(reg, val)    WRITE_REGISTER_ULONG((PULONG)&reg, (ULONG)(val))

/**
 * Puts hardware to/from D0i3 (internal) power state
 *
 * @regs                Address of Registers in MMIO space
 * @powerOff            flag indicating transition direction
 */
NTSTATUS
HwPowerTransition(
    _In_    P_HW_REGS   regs,
    _In_    BOOLEAN     powerOff);

/**
 * Verifies if power transition command is completed or not int progress
 *
 * @regs                hardware registers MMIO address
 * @d0i3                hardware pm control register value
 *
 * @return  TRUE if no pm transition is active, FALSE otherwise
 */
BOOLEAN
HwPowerTransitionVerify(
    _In_    P_HW_REGS   regs,
    _Inout_ D0I3_CTRL*  d0i3);

/******************************************************************************
 * Public Methods
 ******************************************************************************/

ULONG
HwReadReg(
    _In_    PVOID   reg,
    _In_    ULONG   offset)
{
    ULONG value = READ_REGISTER_ULONG((PULONG)((PUCHAR)reg + offset));
    Trace(TLV, T_REG, "%!FUNC! addr:%p, val:%X", (PUCHAR)reg + offset, value);
    return value;
}

VOID
HwWriteReg(
    _In_    PVOID   reg,
    _In_    ULONG   offset,
    _In_    ULONG   value)
{
    Trace(TLV, T_REG, "%!FUNC! addr:%p, val:%X", (PUCHAR)reg + offset, value);
    WRITE_REGISTER_ULONG((PULONG)((PUCHAR)reg + offset), value);
    return;
}

VOID
HwPrepareMmuConfig(
    _In_    PMODEL_CTX  modelCtx,
    _In_    UINT32      length)
{
    PMMU_CONFIG mmu;    // mmu config link
    P_PT_DIR    ptDir;  // page table directory
    ULONG       i;      // page table iterator

    TraceEntry(TLI, T_ENT);

    PDESCRIPTOR descVA = modelCtx->desc.va;
    mmu = &descVA->mmu_config;
    // mark descriptor mmu config data 'dirty'
    RtlFillMemory(mmu, sizeof(MMU_CONFIG), 0xff);
    // populate mmu addresses
    mmu->vamaxaddr = length - 1;
    ptDir = modelCtx->ptDir;
    for (i = 0; i < modelCtx->pageTableCount && i < PT_DIR_SIZE; ++i)
    {
        mmu->pagdir_n[i] = (UINT32)(ptDir[i].commBuffLa.QuadPart/PAGE_SIZE);
    }
}

UINT8
HwReadInBuffSize(
    _In_    P_HW_REGS   regs)
{
    UINT8 value = (UINT8)_READ(regs->ibuffs);
    Trace(TLV, T_REG, "%!FUNC! IBUFFS: %u", value);
    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    return value;
}

VOID
HwMapMemory(
    _In_    PMMU_CONFIG     mmuConfigDst,
    _In_    PMMU_CONFIG     mmuConfigSrc)
{
    TraceEntry(TLI, T_ENT);
    // copy prepared config to base descriptor
    memcpy_s(mmuConfigDst, sizeof(MMU_CONFIG), mmuConfigSrc, sizeof(MMU_CONFIG));
}

VOID
HwUnmapMemory(
    _In_    PMMU_CONFIG     mmuConfig)
{
    TraceEntry(TLI, T_ENT);
    // mark descriptor mmu config data 'dirty' if current app is active
    RtlFillMemory(mmuConfig, sizeof(MMU_CONFIG), 0xff);
}

static
CTRL_REG
HwGetCtrlReg(
    P_HW_REGS           regs)
{
    CTRL_REG ctrl;

    ctrl._dword = _READ(regs->ctrl);
#ifdef _DEBUG
    Trace(TLV, T_REG, "%!FUNC!: CTRL Reg entry-value %08X", ctrl._dword);
#endif
    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    return ctrl;
}

static
VOID
HwStart(
    _In_    P_HW_REGS       regs,
    _In_    PGNA_CALC_IN    input)
{
    CTRL_REG ctrl;                  // control register value

    TraceEntry(TLI, T_ENT);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    ctrl = HwGetCtrlReg(regs);
    ctrl.start_accel = 1;
    ctrl.compl_int_en = 1;
    ctrl.err_int_en = 1;
    ctrl.comp_stats_en = input->hwPerfEncoding & 0xF;
    ctrl.active_list_en = input->ctrlFlags.activeListOn;
    ctrl.gna_mode = input->ctrlFlags.gnaMode;
    _WRITE(regs->ctrl, ctrl._dword);

    getRegs(regs); // debug register dump, necessary to satisfy verbose test timing conditions (e.g. breakpoint tests)
}

VOID
HwInitExecution(
    _In_    P_HW_REGS       regs,
    _In_    ULONG           baseDescriptorLA,
    _In_    PXNN_CONFIG     xnnConfig,
    _In_    PGNA_CALC_IN    input,
    _In_    PDEV_CONFIG     devCfg)
{
    //NTSTATUS sts = STATUS_SUCCESS;
#ifndef ENABLE_LEGACY_INTERRUPTS
    UNREFERENCED_PARAMETER(devCfg);
#endif
    TraceEntry(TLI, T_ENT);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    // wakeup device if needed
    // reset device only if already in D0(i0)
    if (FALSE == HwPowerSwitch(regs, devCfg, POWER_ON))
    {
        HwAbort(regs);
    }

    // copy user provided XNN configuration
    xnnConfig->labase = input->ctrlFlags.layerIndex * XNN_LYR_DSC_SIZE;
    xnnConfig->lacount = (UINT16)input->ctrlFlags.layerCount;

    // start scoring
    _WRITE(regs->desc_base, baseDescriptorLA);
    HwSetInterruptible(devCfg, TRUE);
    HwStart(regs, input);
}

VOID
HwPause(
    _In_    P_HW_REGS   regs)
{
    CTRL_REG ctrl;                  // control register value

    TraceEntry(TLI, T_ENT);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    ctrl = HwGetCtrlReg(regs);
    ctrl.pause_accel = 1;
    ctrl.resume_accel = 0;
    _WRITE(regs->ctrl, ctrl._dword);
}

VOID
HwResume(
    _In_    P_HW_REGS   regs)
{
    CTRL_REG ctrl;                  // control register value

    TraceEntry(TLI, T_ENT);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    ctrl = HwGetCtrlReg(regs);
    ctrl.pause_accel = 0;
    ctrl.resume_accel = 1;
    _WRITE(regs->ctrl, ctrl._dword);
}

VOID
HwAbort(
    _In_    P_HW_REGS   regs)
{
    CTRL_REG ctrl;                  // control register value
    ULONG    i = 0;
    ULONG    sts;

    TraceEntry(TLI, T_ENT);
    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    ctrl = HwGetCtrlReg(regs);
    ctrl.abort_clr_accel = 1;
    _WRITE(regs->ctrl, ctrl._dword);
    // verify if hw is aborted
    sts = 1 & _READ(regs->ctrl);
    while (i < CMD_CMPLT_TIMEOUT && 0 != sts)
    {
        _mm_pause();
        sts = 1 & _READ(regs->ctrl);
        i++;
    };
}

status_t
HwGetIntStatus(
    _In_    PDEV_CTX    devCtx,
    _In_    P_HW_REGS   regs)
{
    STS_REG  stsReg;                // HW status register value
    CTRL_REG ctrlReg;               // HW Control register value
    status_t status = GNA_DEVICEBUSY;// tmp request status
    UNREFERENCED_PARAMETER(devCtx);

    EventWriteHwRegisterRead(NULL, __FUNCTION__);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    getRegs(regs);

    // Read GMMSTS
    stsReg._dword = _READ(regs->sts);
    // check completion status
    if (STS_COMPLETED_FLAG & stsReg._dword) // test if HW has completed scoring
    {
        status = GNA_SUCCESS;
        Trace(TLI, T_QUE, "QueueCompleteExec Scoring Completed");
        if (stsReg._dword & STS_SATURATION_FLAG)  // test if HW has completed scoring with saturation
        {
            // clear 'Score has reached the saturation' flag in GMMSTS
            _WRITE(regs->sts, STS_SATURATION_FLAG);
            status = GNA_SSATURATE;
            Trace(TLW, T_QUE, "%!FUNC! WARNING: Score has reached the saturation!");
        }
    }
    // check warning states
    if (STS_BPPASUE_FLAG & stsReg._dword)
    {
        ctrlReg._dword = _READ(regs->ctrl);
        // clear 'Break Point Pause Interrupt Enable' flag in GMMCTRL and Break Point Setup' registers
        _WRITE(regs->ctrl, ctrlReg._dword & (~CTRL_BPPASUE_INT_EN_FLAG));
        _WRITE(regs->bp._dword[0], 0);
        _WRITE(regs->bp._dword[1], 0);
        // GMM Break Point Pause
        status = GNA_BREAKPOINTPAUSE;
        HwSetInterruptible(&devCtx->cfg, TRUE);
        Trace(TLW, T_QUE, "%!FUNC! WARNING: Scoring paused by breakpoint!");
    }
    if (STS_OUTBUFFULL_FLAG & stsReg._dword)
    {
        _WRITE(regs->sts, STS_OUTBUFFULL_FLAG);
        Trace(TLW, T_QUE, "%!FUNC! WARNING: Output buffer full flag is set!");
    }
    // check error statuses
    if (HW_ERR_FLAGS & stsReg._dword)
    {
        if (STS_MMUREQERR_FLAG   & stsReg._dword) status = GNA_MMUREQERR;
        if (STS_DMAREQERR_FLAG   & stsReg._dword) status = GNA_DMAREQERR;
        if (STS_UNEXPCOMPL_FLAG  & stsReg._dword) status = GNA_UNEXPCOMPL;
        if (STS_VA_OOR_FLAG      & stsReg._dword) status = GNA_VAOUTOFRANGE;
        if (STS_PARAM_OOR_FLAG   & stsReg._dword) status = GNA_PARAMETEROUTOFRANGE;
        Trace(TLE, T_QUE, "%!FUNC! ERROR: Device reported error: 0x%X", stsReg._dword);
    }

    // if request is timeouted and hw is in live loop
    WdfSpinLockAcquire(devCtx->req.reqLock);
    if (GNA_DEVICEBUSY == status && TRUE == devCtx->req.timeouted)
    {
        Trace(TLE, T_QUE, "%!FUNC! CRITICAL ERROR, device hung in live loop, reseting.");
        devCtx->req.timeouted = FALSE;
        WdfSpinLockRelease(devCtx->req.reqLock);
        return GNA_ERR_DEV_FAILURE;
    }
    WdfSpinLockRelease(devCtx->req.reqLock);

    // if no status change, interrupt is warning or other unsupported type
    if (GNA_DEVICEBUSY == status)
    {
        Trace(TLW, T_QUE, "%!FUNC! WARNING: Scoring NOT COMPLETED! Continue scoring.");
        Trace(TLW, T_QUE, "%!FUNC! WARNING: HW Status REG: 0x%X", stsReg._dword);
    }

    return status;
}

#ifdef ENABLE_LEGACY_INTERRUPTS

VOID HwSetInterruptible(
    _In_    PDEV_CONFIG     devCfg,
    _In_    BOOLEAN         enabled)
{
    WdfInterruptAcquireLock(devCfg->interrupt);
    devCfg->interruptible = enabled;
    WdfInterruptReleaseLock(devCfg->interrupt);
}

#endif // ENABLE_LEGACY_INTERRUPTS

BOOLEAN
HwPowerSwitch(
    _In_    P_HW_REGS       regs,
    _In_    PDEV_CONFIG devCfg,
    _In_    BOOLEAN         powerOff)
{
    TraceEntry(TLI, T_ENT);
    // power off device if needed
    WdfSpinLockAcquire(devCfg->pwrLock);
    if (powerOff != devCfg->d0i3Enabled)
    {
        if (STATUS_SUCCESS == HwPowerTransition(regs, powerOff))
        {
            devCfg->d0i3Enabled = powerOff;
            WdfSpinLockRelease(devCfg->pwrLock);
            Trace(TLV, T_REG, "%!FUNC! Power transition (to pwr=%u) complete", (UINT32)(!powerOff));
            return TRUE;
        }
    }
    WdfSpinLockRelease(devCfg->pwrLock);
    Trace(TLV, T_REG, "%!FUNC! Power transition (to pwr=%u) NOT done", (UINT32)(!powerOff));
    return FALSE;
}

/******************************************************************************
 * Private Methods
 ******************************************************************************/

#ifdef _DEBUG

VOID
getRegs(
    _In_    P_HW_REGS   regs)
{
    HW_REGS dump;
    UINT32  i;

    RtlZeroMemory(&dump, sizeof(dump));
    for (i = 0x80; i < 0x118; i += 0x04)
    {
        dump._dword[i / 0x04] = _READ(regs->_dword[i / 0x04]);
    }
    for (i = 0x80; i < 0x118; i += 0x04)
    {
        Trace(TLV, T_QUE, "REG[%04X]\t = 0x%08X", i, dump._dword[i / 0x04]);
    }

    EventWriteHwRegisterRead(NULL, __FUNCTION__);
}

#endif // _DEBUG

NTSTATUS
HwPowerTransition(
    _In_    P_HW_REGS   regs,
    _In_    BOOLEAN     powerOff)
{
    D0I3_CTRL   d0i3 = {0};

    TraceEntry(TLI, T_ENT);
    EventWriteHwRegisterWrite(NULL, __FUNCTION__);

    if (FALSE == HwPowerTransitionVerify(regs, &d0i3))
    {
        Trace(TLW, T_REG, "%!FUNC! WARNING: Previous D0i3 transaction not completed.");
        return STATUS_INVALID_STATE_TRANSITION;
    }
    if (powerOff)
    {
        HwAbort(regs);
    }
    // set d0i3 transition
    d0i3.d0i3 = (UINT32)powerOff;
    d0i3.interrupt_req = 0;
    d0i3.restore_required = 0;
    _WRITE(regs->doi3, d0i3._dword);
    // wait for transition completion
    if (FALSE == HwPowerTransitionVerify(regs, &d0i3))
    {
        Trace(TLW, T_REG, "%!FUNC! WARNING: Current D0i3 transaction not completed.");
        return STATUS_CURRENT_TRANSACTION_NOT_VALID;
    }
    if ((UINT32)powerOff != d0i3.d0i3)
    {
        TraceFailMsg(TLE, T_REG, "%!FUNC! WARNING: Current D0i3 state is invalid.", 0);
        return STATUS_ACPI_POWER_REQUEST_FAILED;
    }
    // transition complete, clear flags
    d0i3.interrupt_req = 0;
    d0i3.restore_required = 0;
    _WRITE(regs->doi3, d0i3._dword);
    return STATUS_SUCCESS;
}

BOOLEAN
HwPowerTransitionVerify(
    _In_    P_HW_REGS   regs,
    _Inout_ D0I3_CTRL*  d0i3)
{
    ULONG   i = 0;

    EventWriteHwRegisterRead(NULL, __FUNCTION__);

    d0i3->_dword = _READ(regs->doi3);
    // verify no pm command is in progress
    while (i < CMD_CMPLT_TIMEOUT && 0 != d0i3->cmd_in_progress)
    {
        _mm_pause();
        d0i3->_dword = _READ(regs->doi3);
        i++;
    };
    return (0 == d0i3->cmd_in_progress); // true = pm cmd completed if 0
}
