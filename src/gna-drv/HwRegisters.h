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

/******************************************************************************
 *
 * GNA MMIO Registers definitions
 *
 *****************************************************************************/

#if !defined(_GNA_REGS_H)
#define _GNA_REGS_H

#pragma warning(disable:4201)

#include "HwDescriptors.h"

/**
 *  Register masks
 */
#define STS_SATURATION_FLAG      0x20000 // WARNING: score has reached the saturation, MUST CLEAR
#define STS_OUTBUFFULL_FLAG      0x10000 // WARNING: hw output buffer is currently full, MUST CLEAR
#define STS_PARAM_OOR_FLAG       0x100   // ERROR: hw parameter out of range
#define STS_VA_OOR_FLAG          0x80    // ERROR: VA out of range
#define STS_UNEXPCOMPL_FLAG      0x40    // ERROR: PCIe error: unexpected completion
#define STS_DMAREQERR_FLAG       0x20    // ERROR: PCIe error: DMA req
#define STS_MMUREQERR_FLAG       0x10    // ERROR: PCIe error: MMU req
#define STS_STATVALID_FLAG       0x08    // compute statistics valid
#define STS_SDTPASUE_FLAG        0x04    // suspended due to pause
#define STS_BPPASUE_FLAG         0x02    // suspended breakpoint match
#define STS_COMPLETED_FLAG       0x01    // scoring completed flag
#define CTRL_BPPASUE_INT_EN_FLAG 0x200
#define PC_REG_SATURATED         0xffffffff
#define HW_ERR_FLAGS (STS_PARAM_OOR_FLAG | STS_VA_OOR_FLAG | STS_UNEXPCOMPL_FLAG | STS_DMAREQERR_FLAG | STS_MMUREQERR_FLAG)
#define HW_INT_FLAGS (STS_COMPLETED_FLAG | STS_BPPASUE_FLAG | STS_SATURATION_FLAG | HW_ERR_FLAGS)

/**
 * Status Register.
 */
typedef union _STS_REG
{
    struct 
    {
    UINT32  scr_completed       :1; // 00:00 ROV - scoring completed
    UINT32  susp_bp_match       :1; // 01:01 ROV - suspended breakpoint match
    UINT32  susp_pause          :1; // 02:02 ROV - suspended due to pause
    UINT32  comp_stats_valid    :1; // 03:03 ROV - compute statistics valid
    UINT32  pcie_mmu_err        :1; // 04:04 ROV - PCIe error: MMU req
    UINT32  pcie_dma_err        :1; // 05:05 ROV - PCIe error: DMA req
    UINT32  pcie_ucomp_err      :1; // 06:06 ROV - PCIe error: unexpected completion
    UINT32  va_oor_err          :1; // 07:07 ROV - VA out of range
    UINT32  hw_par_oor_err      :1; // 08:08 ROV - parameter out of range
    UINT32  __res_15_9          :7; // 09:15 RO  - reserved 
    UINT32  hw_out_full         :1; // 16:16 RWC - output buffer is currently full
    UINT32  score_saturated     :1; // 17:17 RWC - score has reached the saturation
    UINT32  __res_23_18         :6; // 18:23 RWC - reserved
    UINT32  __res_31_24         :6; // 24:31 RO - reserved
    };
    UINT32  _dword;                 // value of whole register

} STS_REG;                          // Status Register

static_assert(4 == sizeof(STS_REG), "Invalid size of STS_REG");

/**
 * Control Register.
 */
typedef union _CTRL_REG
{
    struct
    {
    UINT32  start_accel         :1; // 00:00 RWS - start accelerator
    UINT32  active_list_en      :1; // 01:01 RW  - active list enable
    UINT32  abort_clr_accel     :1; // 02:02 RWO - abort/clear accelerator
    UINT32  pause_accel         :1; // 03:03 RWS - pause execution
    UINT32  resume_accel        :1; // 04:04 RWS - resume execution
    UINT32  gna_mode            :2; // 05:06 RW  - GNA operation mode (0:GMM, 1:xNN)
    UINT32  __res_7             :1; // 07:07 RO  - reserved 
    UINT32  compl_int_en        :1; // 08:08 RW  - completion interrupt enable
    UINT32  bp_pause_int_en     :1; // 09:09 RW  - breakpoint pause interrupt enable
    UINT32  err_int_en          :1; // 10:10 RW  - error interrupt enable
    UINT32  __res_11            :1; // 11:11 RO  - reserved
    UINT32  comp_stats_en       :4; // 12:15 RW  - compute statistics enable
    UINT32  pm_ovr_power_on     :1; // 16:16 RW  - pwr mgmt override power on
    UINT32  pm_ovr_clock_on     :1; // 17:17 RW  - pwr mgmt override force clck on
    UINT32  pm_quite_idle_dis   :1; // 18:18 RW  - pwr mgmt quite-idle disable // NOTE: available on SKL ONLY.
    UINT32  __res_23_19         :5; // 19:23 ROV - reserved 
    UINT32  __res_31_24         :8; // 24:31 ROV - reserved 
    };
    UINT32 _dword;                  // value of whole register

} CTRL_REG;                         // Control Register

static_assert(4 == sizeof(CTRL_REG), "Invalid size of CTRL_REG");

/**
 * Management Control.
 */
typedef union _MCTRL_REG
{
    struct
    {
    UINT32  max_outs_trans      :4; // 00:03 RW - max outstanding transaction control
    UINT32  __res_7_4           :4; // 04:07 RO - reserved 
    UINT32  sa_hi_freq_req      :1; // 08:08 RW - System agent high frequency required
    UINT32  __res_31_9          :23;// 09:31 RO - reserved 
    };
    UINT32  _dword;                 // value of whole register

} MCTRL_REG;                        // Management Control

static_assert(4 == sizeof(MCTRL_REG), "Invalid size of MCTRL_REG");

typedef UINT32 PTC_REG;             // 00:31 ROV - Performance Total Cycles

typedef UINT32 PSC_REG;             // 00:31 ROV - Performance Stall Cycles

/**
 * Internal State Index.
 */
typedef union _ISI_REG
{
    struct
    {
    UINT32  int_state_idx       :11;// 00:10 RW - index of internal GNA module status
    UINT32  __res_31_11         :21;// 11:31 RO - reserved
    };
    UINT32  _dword;                 // value of whole register

} ISI_REG;                          // Internal State Index.

static_assert(4 == sizeof(ISI_REG), "Invalid size of ISI_REG");

/*
 * Internal State Value
 */
typedef union _ISV_REG
{
    struct
    {
    UINT32  low;                    // 00:31 ROV - low status bits
    UINT32  hi;                     // 31:63 ROV - hi status bits
    };
    UINT64  _qword;                 // value of whole register

} ISV_REG;                          // Internal State Value

static_assert(8 == sizeof(ISV_REG), "Invalid size of ISV_REG");

/*
 * Break Point Setup
 */
typedef union _BP_SETUP
{
    // GMM Mode Break point setup
    struct
    {
    UINT32  mix_comp_num        :13;// 00:12 RW - Mixture comp. number
    UINT32  states              :19;// 13:31 RW - GMM state
    UINT32  active_index        :19;// 32:50 RW - Active index
    UINT32  __res_63_51         :13;// 51:63 RO - reserved

    } gmm;                          // GMM Mode Break point setup

    // xNN Mode Break point setup
    struct
    {
    UINT32  output_num          :16;// 00:15 RW - output number
    UINT32  input_num           : 8;// 16:23 RW - input number
    UINT32  in_iter_num         : 8;// 24:31 RW - input iteration number
    UINT32  group_num           : 3;// 32:34 RW - group number
    UINT32  layer_num           :10;// 35:44 RW - layer number
    UINT32  __res_63_45         :18;// 45:62 RO - (reserved)
    UINT32  xnn_debug_en        : 1;// 63:63 RW - (18b reserved)

    } xnn;                          // xNN Mode Break point setup

    UINT32  _dword[2];              // value of whole register

} BP_SETUP;                         // Break Point Setup for GMM MODE ONLY

static_assert(8 == sizeof(BP_SETUP), "Invalid size of BP_SETUP");

/*
 * D0i3 control
 */
typedef union _D0I3_CTRL
{
    struct
    {
    UINT32  cmd_in_progress     :1; // 00:00 ROV - command in progress
    UINT32  interrupt_req       :1; // 01:01 RW - interrupt request 0:disabled, 1:enabled
    UINT32  d0i3                :1; // 02:02 RW - D0i3, 1:set to D0i3, 0:return to D0i0
    UINT32  restore_required    :1; // 03:03 RW1C - Restore required
    UINT32  __res_31_04         :28;// 04:31 RO - (28b reserved)
    };
    UINT32  _dword;                 // value of whole register

} D0I3_CTRL;                        // D0i3 control register

static_assert(4 == sizeof(D0I3_CTRL), "Invalid size of D0I3_CTRL");

typedef UINT32 DESC_BASE_REG;       // GNA Base Desriptor Address Register

// TODO: verify if there is corrected size information in HAS
/*
 * Hw Information
 */
typedef union _IBUFFS
{
    struct
    {
    UINT32  in_buff_size        :8; // 00:07 RO - input buffer size in KB
    UINT32  __res_31_08         :24;// 08:31 RO - (24b reserved)
    };
    UINT32  _dword;                 // value of whole register

} IBUFFS;                           // Hw Information

static_assert(4 == sizeof(IBUFFS), "Invalid size of IBUFFS");

/*
 * Access control RW SAI
 */
typedef union _SAI
{
    UINT32  _dword[2];              // value of whole register

} SAI;                              // Access control RW SAI

static_assert(8 == sizeof(SAI), "Invalid size of SAI");

/*
 * Access control SAI value
 */
typedef union _SAIV
{
    UINT32  _dword[2];              // value of whole register

} SAIV;                             // Access control SAI value

static_assert(8 == sizeof(SAIV), "Invalid size of SAIV");

/**
 * GNA Configuration Control and Status register
 *
 * Offset:  0x0000
 * Size:    0x1000 (4096B)
 * See:     HAS Section 6.2.2.1
 * Note:    Specifies whole GNA HW configuration
 *          Memory mapped register.
 */
typedef union _HW_REGS
{
    struct
    {
    __1B_RES        _gmm_shdw[128]; // 0000 - 007F (128 B) - RO - GMM Shadow registers
    STS_REG         sts;            // 0080 - 0083 (004 B) - RW - Status
    CTRL_REG        ctrl;           // 0084 - 0087 (004 B) - RW - Control
    MCTRL_REG       mgmt;           // 0088 - 008B (004 B) - RW - Management control
    PTC_REG         ptc;            // 008C - 008F (004 B) - RO - Performance Total Cycles
    PSC_REG         psc;            // 0090 - 0093 (004 B) - RO - Performance Stall Cycles
    ISI_REG         isi;            // 0094 - 0097 (004 B) - RW - Internal State Index
    ISV_REG         isv;            // 0098 - 009F (008 B) - RO - Internal State Value
    BP_SETUP        bp;             // 00A0 - 00A7 (008 B) - RW - Break point setup
    D0I3_CTRL       doi3;           // 00A8 - 00AB (004 B) - RW - D0i3 Control
    __1B_RES        __res_00A9[4];  // 00AC - 00AF (004 B reserved) - // TODO: missing in has
    DESC_BASE_REG   desc_base;      // 00B0 - 00B3 (004 B) - RW - Descriptor base address
    IBUFFS          ibuffs;         // 00B4 - 00B7 (004 B) - RW - Information - TODO: size error in has
    __1B_RES        __res_00B8[72]; // 00B8 - 00FF (072 B reserved)
    // 256 B
    SAI             sai1;           // 0100 - 0107 (008 B) - RW - SAI1
    SAI             sai2;           // 0108 - 010F (008 B) - RW - SAI2
    SAIV            saiv;           // 0110 - 0117 (008 B) - RW - SAI 232alue
    __1B_RES        __res_0118[232];// 0118 - 01FF (232 B reserved)
    // 512B
    __1B_RES        __res_0200[1536];//0200 - 07FF (1536 B reserved)
    // 2048B
    __1B_RES        _wp_regs[2048]; // 0800 - 0FFF (2048 B) - Wrapper passed registers block?
    // 4096B
    };
    UINT32  _dword[1024];                 // value of whole register

} HW_REGS, *P_HW_REGS;              //  Configuration Control and Status register

static_assert(4096 == sizeof(HW_REGS), "Invalid size of HW_REGS");

#endif // _GNA_REGS_H
