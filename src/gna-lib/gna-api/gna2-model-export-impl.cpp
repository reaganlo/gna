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
#include "gna2-model-export-impl.h"

#include "ApiWrapper.h"
#include "Device.h"
#include "DeviceManager.h"
#include "ModelExportConfig.h"

#include "gna2-common-impl.h"
using namespace GNA;

GNA2_API enum Gna2Status Gna2ModelExportConfigCreate(
    Gna2UserAllocator userAllocator,
    uint32_t * const exportConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull((void*)userAllocator);
        Expect::NotNull(exportConfigId);
        *exportConfigId = ModelExportManager::GetManager().AddConfig(userAllocator);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExportConfigRelease(
    uint32_t exportConfigId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        ModelExportManager::GetManager().RemoveConfig(exportConfigId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = ModelExportManager::GetManager().GetConfig(exportConfigId);
        config.SetSource(sourceDeviceIndex, sourceModelId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum Gna2DeviceVersion targetDeviceVersion)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = ModelExportManager::GetManager().GetConfig(exportConfigId);
        config.SetTarget(targetDeviceVersion);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelExport(
    uint32_t exportConfigId,
    enum Gna2ModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& config = ModelExportManager::GetManager().GetConfig(exportConfigId);
        config.Export(componentType, exportBuffer, exportBufferSize);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}