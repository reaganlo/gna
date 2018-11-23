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

#if !defined(_HW_H)
#define _HW_H

#include "Driver.h"
// TODO: split into private/hidden hw access interface and hw logic

/**
 * Power transition direction - put HW to D0i3 (sleep)
 **/
#define     HW_POWER_OFF               TRUE

/**
 * Power transition direction - put HW to D0i0 (active)
 **/
#define     POWER_ON                FALSE

/**
 * Reads hardware register at given address
 *
 * @reg                 Address of Register in MMIO space
 * @offset              Register offset
 */
ULONG
HwReadReg(
    _In_    PVOID       reg,
    _In_    ULONG       offset);

/**
 * Writes value to hardware register at given address
 *
 * @reg                 Address of Registers in MMIO space
 * @offset              Register offset
 * @value               Value to be written into the register
 */
VOID
HwWriteReg(
    _In_    PVOID       reg,
    _In_    ULONG       offset,
    _In_    ULONG       value);

/**
 * Warning: backward compatibility function
 * Prepares hardware mmu config for further copying to hw base descriptor
 *
 * @appCtx              memory mapping data
 * @length              length of mapped buffer
 */
VOID
HwPrepareMmuConfig(
    _In_    PAPP_CTX    appCtx,
    _In_    UINT32      length);

/**
 * Prepares hardware mmu config for further copying to hw base descriptor
 *
 * @appCtx              memory mapping data
 */
VOID
HwPrepareMmuConfig2(
    _In_    PMEMORY_CTX  memoryCtx);

/**
 * Retrieves hardware internal Input buffer size
 *
 * @regs                Address of Registers in MMIO space
 * @return  input buffer size in KB
 */
UINT8
HwReadInBuffSize(
    _In_    P_HW_REGS   regs);

/**
 * Stores memory mapping into the hardware device descriptor
 *
 * @mmuConfigDst        hardware descriptor mmu config buffer (destination)
 * @mmuConfigSrc        prepared mmu config buffer (source)
 */
VOID
HwMapMemory(
    _In_    PMMU_CONFIG mmuConfigDst,
    _In_    PMMU_CONFIG mmuConfigSrc);

/**
 * Wipes out memory mapping from the hardware device descriptor
 *
 * @mmuConfigDst        hardware descriptor mmu config buffer
 */

VOID
HwUnmapMemory(
    _In_    PMMU_CONFIG mmuConfig);

/**
 * Programs the hardware device and starts execution
 *
 * @regs                Address of Registers in MMIO space
 * @config              Pointer to xnn configuration of the model
 * @lyrDscBuffer        Pointer to layer descriptors
 * @input               Input data obtained from user application
 * @baseDescLa          Physical address of base descriptor
 * @devCfg              device config for legacy mode
 */
VOID
HwInitExecution(
    _In_    P_HW_REGS       regs,
    _In_    PVOID           config,
    _In_    PUINT8          lyrDscBuffer,
    _In_    PGNA_CALC_IN    input,
    _In_    ULONG           baseDescriptorLA,
    _In_    PDEV_CONFIG     devCfg);

/**
 * Pause execution hardware device command
 *
 * @regs                Address of Registers in MMIO space
 */
VOID
HwPause(
    _In_    P_HW_REGS   regs);

/**
 * Resume execution hardware device command
 *
 * @regs                Address of Registers in MMIO space
 */
VOID
HwResume(
    _In_    P_HW_REGS   regs);

/**
 * Abort execution hardware device command
 *
 * @regs                Address of Registers in MMIO space
 */
VOID
HwAbort(
    _In_    P_HW_REGS   regs);

/**
 * Gets hardware device status information
 *
 * @devCtx              Device context
 * @regs                Address of Registers in MMIO space
 */
status_t
HwGetIntStatus(
    _In_    PDEV_CTX    devCtx,
    _In_    P_HW_REGS   regs);

/**
 * Switches hardware device interruptible state in LEGACY mode
 *
 * @devCfg              Device config
 * @enabled             TRUE: enable interrupts, FALSE:disable
 */
#ifdef ENABLE_LEGACY_INTERRUPTS

VOID
HwSetInterruptible(
    _In_    PDEV_CONFIG devCfg,
    _In_    BOOLEAN     enabled);

#else // MSI

#define HwSetInterruptible(devCfg, enabled) // disabled in MSI INT mode

#endif // ENABLE_LEGACY_INTERRUPTS

/**
 * Puts hardware to/from D0i3 (internal) power state
 *
 * @regs                Address of Registers in MMIO space
 * @devCfg              Device config
 * @powerOff            flag indicating transition direction
 *                      * TRUE  -> to power off
 *                      * FALSE -> to power on)
 */
BOOLEAN
HwPowerSwitch(
    _In_    P_HW_REGS   regs,
    _In_    PDEV_CONFIG devCfg,
    _In_    BOOLEAN     powerOff);

#endif // _HW_H
