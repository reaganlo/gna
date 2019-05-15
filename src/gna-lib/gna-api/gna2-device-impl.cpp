/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "gna2-device-impl.h"

#include "ApiWrapper.h"
#include "DeviceManager.h"
#include "Expect.h"
#include "Logger.h"

#include "gna2-common-impl.h"

#include <functional>

using namespace GNA;

enum Gna2Status Gna2DeviceGetCount(
    uint32_t * deviceCount)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(deviceCount);
        *deviceCount = DeviceManager::Get().GetDeviceCount();
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceGetVersion(
    uint32_t deviceIndex,
    enum Gna2DeviceVersion * deviceVersion)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(deviceVersion);
        *deviceVersion = DeviceManager::Get().GetDeviceVersion(deviceIndex);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceSetNumberOfThreads(
    uint32_t deviceIndex,
    uint32_t numberOfThreads)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDevice(deviceIndex);
        device.SetNumberOfThreads(numberOfThreads);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceOpen(
    uint32_t deviceIndex)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().OpenDevice(deviceIndex);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

enum Gna2Status Gna2DeviceClose(
    uint32_t deviceIndex)
{
    const std::function<ApiStatus()> command = [&]()
    {
        DeviceManager::Get().CloseDevice(deviceIndex);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

