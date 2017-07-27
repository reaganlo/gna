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

#pragma once


/******************************************************************************
 *
 * Windows Specific Driver interface
 *
 *****************************************************************************/

#ifdef DRIVER
#   include <ntddk.h>
#else
#   include <initguid.h>
#   include <Windows.h>
#endif // !DRIVER

#include "GnaDrvApi.h"

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size

/**
 * Define an Interface Guid so that app can find the device and talk to it.
 */

// {8113B324-9F9B-4B9F-BF55-1342A58593DC}
DEFINE_GUID(GUID_DEVINTERFACE_GNA_DRV,
    0x8113b324, 0x9f9b, 0x4b9f, 0xbf, 0x55, 0x13, 0x42, 0xa5, 0x85, 0x93, 0xdc);

// {608D09B8-41BC-4079-A040-1EE3F48483DD}
DEFINE_GUID(GUID_DEVINTERFACE_GMM_DRV,
    0x608d09b8, 0x41bc, 0x4079, 0xa0, 0x40, 0x1e, 0xe3, 0xf4, 0x84, 0x83, 0xdd);

/******************************************************************************
 *
 * Driver IOCTL interface
 *
 *****************************************************************************/

#define FILE_DEVICE_PCI_GNA 0x8000
#define GNA_IOCTL_MEM_MAP   CTL_CODE(FILE_DEVICE_PCI_GNA, 0x900, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define GNA_IOCTL_MEM_UNMAP CTL_CODE(FILE_DEVICE_PCI_GNA, 0x901, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_CPBLTS    CTL_CODE(FILE_DEVICE_PCI_GNA, 0x902, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_NOTIFY    CTL_CODE(FILE_DEVICE_PCI_GNA, 0x907, METHOD_NEITHER, FILE_ANY_ACCESS)

#pragma pack ()

