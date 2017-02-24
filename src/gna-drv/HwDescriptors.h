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

#if !defined(_GNA_DESC_H)
#define _GNA_DESC_H

#include "GnaDrvApi.h"

#pragma warning(disable:4201)

typedef     UINT8       __1B_RES;   // 1B of reserved memory

static_assert(1 == sizeof(UINT8), "Invalid size of UINT8");
static_assert(4 == sizeof(UINT32), "Invalid size of UINT32");

/******************************************************************************
 * MMU Configuration Fields (kept in driver private memory)
 ******************************************************************************/

/**
 * MMU - Virtual Address Max Address
 *
 * Offset:   0x0200
 * Size:     0x04 B
 * See:      HAS Section 5.4.4.1
 * Note:     Specifies the maximum virtual address the DMA can legally access.
 *           Should point to the last valid address.
 */
typedef     UINT32      VAMAXADDR;

/**
 * MMU - Page Directory Entry
 *
 * Offset:   0x0210
 * Size:     0x0100 B
 * See:      HAS Section 5.4.4.1
 * Note:     Page Directory consists of 64 Page Directory Entries.
 *           Each entry specifies physical address pointer to 1 Page Table.
 */
typedef     UINT32      PAGDIR_N;

/******************************************************************************
 *
 * GNA Descriptors definitions (kept in driver private memory)
 *
 ******************************************************************************/

/**
 * GNA XNN configuration descriptor
 *
 * Offset:  0x100
 * Size:    0x100 = 256B
 * See:     HAS Section 5.4.3.8
 * Note:    Specifies xnn configuration
 */
typedef union _XNN_CONFIG
{
    struct 
    {
        UINT32 labase;
        UINT32 lacount;
    };
    UINT8   _byte[256];
} XNN_CONFIG, *PXNN_CONFIG;

/**
 * MMU Setup & Directory
 * 
 * Offset:   0x0200
 * Size:     0x0110 // TODO: verify has is correct
 * See:      HAS Section 5.4.4     
 * Note:     Specifies user pinned memory parameters
 */
typedef union _MMU_CONFIG
{
    struct
    {
    VAMAXADDR   vamaxaddr;          // 0200 - 0203 - Virtual address max address
    __1B_RES    __res_204[12];      // 0204 - 020F (12B reserved)
    PAGDIR_N    pagdir_n[PT_DIR_SIZE];// 0210 - 030F - Page directory entries (4 x PT_DIR_SIZE)
    };
    UINT8      _byte[272];          // value of whole register

} MMU_CONFIG, *PMMU_CONFIG;         // MMU Setup & Directory

static_assert(272 == sizeof(MMU_CONFIG), "Invalid size of MMU_CONFIG");

/**
 * GNA Base Descriptor
 *
 * Size:    0x0310 = 784B
 * See:     HAS Section 5.4.1
 * Note:    Specifies base accelerator parameters
 *          Read-only for GNA hardware
 */
typedef union _DESCRIPTOR
{
    struct
    {
    __1B_RES    __res_0000[256];    // 0000 - 00FF - (256B reserved)
    XNN_CONFIG  xnn_config;      // 0100 - 01FF - Whole GNA Configuration, 256B
    MMU_CONFIG  mmu_config;         // 0200 - 030F - MMU setup and directory
    };
    UINT8      _byte[784];          // value of whole register

} DESCRIPTOR, *PDESCRIPTOR;         // GNA Base Descriptor

static_assert(784 == sizeof(DESCRIPTOR), "Invalid size of DESCRIPTOR");

/**
 * GNA Device private configuration space size
 *  Configuration space is additional memory in mapped memory space
 *          used for device descriptor configuration
 *  NOTE:   real size of 784B is used, but minimum allocation size
 *          for physical contiguous memory is PAGE_SIZE
 */
#define PRV_CFG_SIZE    PAGE_SIZE

#endif // _GNA_DESC_H
