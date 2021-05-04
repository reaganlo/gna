//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material contains trade secrets and proprietary
// and confidential information of Intel or its suppliers and licensors.
// The Material is protected by worldwide copyright and trade secret laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed in any way
// without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//*****************************************************************************
#pragma once

#ifdef _WIN32
#include "WindowsHardwareSelfTest.h"
typedef class WindowsGnaSelfTestHardwareStatus MultiOsGnaSelfTestHardwareStatus;
#elif __linux__
#include "LinuxHardwareSelfTest.h"
typedef class LinuxGnaSelfTestHardwareStatus MultiOsGnaSelfTestHardwareStatus;
#endif

// Currently only Android compile-time configuration
void PrintSystemInfo()
{
#ifdef __ANDROID__
    logger.Verbose("This application was build for Android OS\n");
    logger.Verbose("Preprocessor definition __ANDROID__     = %d\n",int(__ANDROID__ ));
#endif
#ifdef __ANDROID_API__
    logger.Verbose("Preprocessor definition __ANDROID_API__ = %d\n",int(__ANDROID_API__));
#endif
}
