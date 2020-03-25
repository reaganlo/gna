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

#include "ModelExportConfig.h"

#include "DeviceManager.h"
#include "GnaException.h"

#include "gna2-model-export-impl.h"
#include "gna2-model-suecreek-header.h"

#include <map>
#include <cstdint>

using namespace GNA;

ModelExportConfig::ModelExportConfig(Gna2UserAllocator userAllocator) : allocator{ userAllocator }
{
    Expect::NotNull((void *)userAllocator);
}

void ModelExportConfig::Export(Gna2ModelExportComponent componentType, void ** exportBuffer, uint32_t * exportBufferSize)
{
    Expect::NotNull(exportBufferSize);
    Expect::NotNull(exportBuffer);

    ValidateState();

    *exportBuffer = nullptr;
    *exportBufferSize = 0;
    Gna2Status status;
    auto& device = DeviceManager::Get().GetDevice(sourceDeviceId);
    if (componentType == Gna2ModelExportComponentLegacySueCreekHeader)
    {
        *exportBufferSize = sizeof(Gna2ModelSueCreekHeader);
        const auto header = static_cast<Gna2ModelSueCreekHeader *>(allocator(*exportBufferSize));
        *exportBuffer = header;
        const auto dump = device.Dump(sourceModelId,
            reinterpret_cast<intel_gna_model_header*>(header), &status, privateAllocator);
        privateDeAllocator(dump);
        return;
    }

    if (componentType == Gna2ModelExportComponentLegacySueCreekDump)
    {
        intel_gna_model_header header;
        *exportBuffer = device.Dump(sourceModelId, &header, &status, allocator);
        *exportBufferSize = header.model_size;
        return;
    }

    if (componentType == Gna2ModelExportComponentLayerDescriptors)
    {
        Expect::True(targetDeviceVersion == Gna2DeviceVersionEmbedded3_0, Gna2StatusAccelerationModeNotSupported);
        device.DumpLdNoMMu(sourceModelId, allocator, *exportBuffer, *exportBufferSize);
        return;
    }

    throw GnaException(Gna2StatusNotImplemented);
}

void ModelExportConfig::SetSource(uint32_t deviceId, uint32_t modelId)
{
    sourceDeviceId = deviceId;
    sourceModelId = modelId;
}

void ModelExportConfig::SetTarget(Gna2DeviceVersion version)
{
    targetDeviceVersion = version;
}

void ModelExportConfig::ValidateState() const
{
    Expect::NotNull((void *)allocator);
    //TODO:3:Consider adding ~Gna2StatusInvalidState/NotInitialized
    Expect::True(sourceDeviceId != Gna2DisabledU32, Gna2StatusIdentifierInvalid);
    Expect::True(sourceModelId != Gna2DisabledU32, Gna2StatusIdentifierInvalid);
    // TODO:3: remove when toolchain is consistent with API.
    uint32_t const legacySueCreekVersionNumber = 0xFFFF0001;
    auto const is1x0Embedded = Gna2DeviceVersionEmbedded1_0 == targetDeviceVersion
    || legacySueCreekVersionNumber == static_cast<uint32_t>(targetDeviceVersion);
    auto const is3x0Embedded = targetDeviceVersion == Gna2DeviceVersionEmbedded3_0;
    //TODO:3:Remove when other devices supported
    Expect::True(is1x0Embedded || is3x0Embedded, Gna2StatusAccelerationModeNotSupported);
}

inline void * ModelExportConfig::privateAllocator(uint32_t size)
{
    return _mm_malloc(size, 4096);
}

inline void ModelExportConfig::privateDeAllocator(void * ptr)
{
    _mm_free(ptr);
}

ModelExportManager & ModelExportManager::Get()
{
    static ModelExportManager globalManager;
    return globalManager;
}

uint32_t ModelExportManager::AddConfig(Gna2UserAllocator userAllocator)
{
    auto idCreated = configCount++;
    allConfigs.emplace(idCreated, ModelExportConfig{ userAllocator });
    return idCreated;
}

void ModelExportManager::RemoveConfig(uint32_t id)
{
    allConfigs.erase(id);
}

ModelExportConfig & ModelExportManager::GetConfig(uint32_t exportConfigId)
{
    auto found = allConfigs.find(exportConfigId);
    if (found == allConfigs.end())
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
    return found->second;
}
