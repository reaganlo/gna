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

#if !defined (_DEVICE_H)
#define _DEVICE_H

#include "Driver.h"

/**
 * Device D0 power state entry Event handler
 */
EVT_WDF_DEVICE_D0_ENTRY
DeviceD0EntryEvnt;

/**
 * Device D0 power state exit Event handler
 */
EVT_WDF_DEVICE_D0_EXIT
DeviceD0ExitEvnt;

/**
 * File create (app opening device) Event handler
 */
EVT_WDF_DEVICE_FILE_CREATE
FileCreateEvnt;

/**
 * File close (app closed device) Event handler
 */
EVT_WDF_FILE_CLOSE
FileCloseEvnt;

/**
 * File cleanup (app closing device) Event handler
 */
EVT_WDF_FILE_CLEANUP
FileCleanupEvnt;

#endif // _DEVICE_H
