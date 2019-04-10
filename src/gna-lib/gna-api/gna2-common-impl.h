/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#ifndef __GNA2_COMMON_IMPL_H
#define __GNA2_COMMON_IMPL_H

#include <gna2-common-api.h>

#include <stdint.h>
#if defined(__GNUC__)
#define UNREFERENCED_PARAMETER(P) ((void)(P))
#else
#include <windows.h>
#endif

namespace GNA
{

typedef enum Gna2DeviceVersion DeviceVersion;

DeviceVersion const DefaultDeviceVersion = GNA2_DEFAULT_DEVICE_VERSION;

typedef enum Gna2Status ApiStatus;

constexpr uint32_t const Gna2DisabledU32 = (uint32_t)GNA2_DISABLED;

constexpr int32_t const Gna2DisabledI32 = (int32_t)GNA2_DISABLED;

constexpr uint32_t const Gna2DefaultU32 = (uint32_t)GNA2_DEFAULT;

constexpr int32_t const Gna2DefaultI32 = (int32_t)GNA2_DEFAULT;

constexpr uint32_t const Gna2NotSupportedU32 = (uint32_t)GNA2_NOT_SUPPORTED;

constexpr int32_t const Gna2NotSupportedI32 = (int32_t)GNA2_NOT_SUPPORTED;

// temporary cast for simultaneous 2 apis usage
#define CAST2_STATUS (ApiStatus)

}

#endif //ifndef __GNA2_COMMON_IMPL_H
