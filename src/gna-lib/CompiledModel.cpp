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

#include "CompiledModel.h"

#include "Macros.h"
#include "Memory.h"
#include "RequestConfiguration.h"
#include "SubModel.h"

#include <functional>

using namespace GNA;

CompiledModel::CompiledModel(
        const gna_model *const userModel,
        const AccelerationDetector& detectorIn,
        const HardwareCapabilities& hwCapabilitiesIn,
        std::vector<std::unique_ptr<Memory>>& memoryObjects) :
    LayerCount{ userModel->nLayers },
    GmmCount{ getGmmCount(userModel) },
    detector { detectorIn },
    hwCapabilities{ hwCapabilitiesIn },
    memoryList { memoryObjects },
    modelMemoryList { },
    softwareModel
    {
        userModel,
        makeValidator(),
        detector.GetSupportedCpuAccelerations()
    }
{
}

void CompiledModel::CopyData(void *address, size_t size) const
{
    auto modelSize = CalculateSize();
    if (size < modelSize)
    {
        throw GnaException{ GNA_ERR_RESOURCES };
    }

    for (const auto &memory : modelMemoryList)
    {
        auto memorySize = memory->GetSize();
        memcpy_s(address, size, memory->GetBuffer(), memorySize);
        size -= memorySize;
        address = static_cast<void *>(
            static_cast<uint8_t *>(address) + memorySize);
    }
}

uint32_t CompiledModel::CalculateSize() const
{
    uint32_t modelSize = 0;
    for (const auto &memory : modelMemoryList)
    {
        modelSize += static_cast<uint32_t>(memory->GetSize());
    }

    return modelSize;
}

void CompiledModel::BuildHardwareModel(DriverInterface &ddi)
{
    const auto& deviceSubmodels = getSubmodels(hwCapabilities);

    auto hasHardwareCompliantLayer =
        !(1 == submodels.size() && SubmodelType::Software == deviceSubmodels.at(0)->Type);
    if(!hasHardwareCompliantLayer)
    {
        Log->Warning("None of model layers is compliant with selected hardware GNA device, "
            "only software processing is available for this model.\n");
    }
    else if (hwCapabilities.IsHardwareSupported())
    {
        hardwareModel = std::make_unique<HardwareModelScorable>(
                                softwareModel.Layers, GmmCount, ddi, hwCapabilities);
    }

    if (hardwareModel)
    {
        hardwareModel->Build(modelMemoryList);
    }
}

const std::vector<std::unique_ptr<Layer>>& CompiledModel::GetLayers() const
{
    return softwareModel.Layers;
}

const Layer* CompiledModel::GetLayer(uint32_t layerIndex) const
{
    try
    {
        return softwareModel.Layers.at(layerIndex).get();
    }
    catch (const std::exception&)
    {
        throw GnaException(XNN_ERR_LYR_CFG);
    }
}

void CompiledModel::InvalidateConfig(gna_request_cfg_id configId, LayerConfiguration *layerConfiguration, uint32_t layerIndex) const
{
    if (hardwareModel)
    {
        hardwareModel->InvalidateConfig(configId);
    }

    auto layer = GetLayer(layerIndex);
    layer->UpdateKernelConfigs(*layerConfiguration);
}

status_t CompiledModel::Score(
    RequestConfiguration& config,
    RequestProfiler *profiler,
    KernelBuffers *buffers)
{
    profilerDTscStart(&profiler->scoring);

    auto saturationCount = uint32_t{0};
    const auto isHardwareEnforced = config.Acceleration.IsHardwareEnforced();
    const auto isSoftwareEnforced = !hwCapabilities.IsHardwareSupported() ||
        (config.Acceleration.IsSoftwareEnforced());

    try
    {
        if (isHardwareEnforced && !hardwareModel)
        {
            return GNA_CPUTYPENOTSUPPORTED;
        }

        if (isHardwareEnforced || config.HasConsistencyMode())
        {
            auto hwCaps = hwCapabilities;
            if (config.HasConsistencyMode()
                    && hwCaps.GetDeviceVersion() != config.GetConsistentDevice())
            {
                hwCaps = HardwareCapabilities(config.GetConsistentDevice());
            }

            const auto& deviceSubmodels = getSubmodels(hwCaps);
            if (deviceSubmodels.front()->Type == Software || deviceSubmodels.size() > 1)
            {
                return GNA_CPUTYPENOTSUPPORTED;
            }
        }

        if (isSoftwareEnforced)
        {
            saturationCount = softwareModel.Score(0, LayerCount, config, profiler, buffers);
        }
        else
        {
            saturationCount = scoreAllSubModels(config, profiler, buffers);
        }
    }
    catch (const GnaException& e)
    {
        return e.getStatus();
    }
    profilerDTscStop(&profiler->scoring);
    profilerDTscStop(&profiler->total);

    return (saturationCount > 0) ? GNA_SSATURATE : GNA_SUCCESS;
}

void CompiledModel::ValidateBuffer(
    std::vector<Memory *> &configMemoryList, Memory *memory) const
{
    if (hardwareModel)
    {
        hardwareModel->ValidateConfigBuffer(configMemoryList, memory);
    }
}

// TODO:3: count buffer use in model to minimize redundant mapping
Memory * CompiledModel::FindBuffer(const void *buffer, size_t bufferSize) const
{
    for(const auto &memory : memoryList)
    {
        auto memoryBuffer = memory->GetBuffer();
        auto memorySize = memory->GetSize();
        if (Expect::InMemoryRange(buffer, bufferSize, memoryBuffer, memorySize))
        {
            return memory.get();
        }
    }

    return nullptr;
}

bool CompiledModel::IsPartOfModel(Memory *memory) const
{
    auto foundIt = std::find(modelMemoryList.cbegin(), modelMemoryList.cend(), memory);
    return foundIt != modelMemoryList.cend();
}

void CompiledModel::AddUniqueMemory(Memory *memory)
{
    if (!IsPartOfModel(memory))
    {
        modelMemoryList.push_back(memory);
    }
}

void CompiledModel::IdentifyBuffer(const void *buffer, size_t bufferSize)
{
    auto memory = FindBuffer(buffer, bufferSize);
    Expect::NotNull(memory, XNN_ERR_INVALID_BUFFER);

    AddUniqueMemory(memory);
}

uint32_t CompiledModel::getGmmCount(const gna_model *const userModel) const
{
    uint32_t gmmCount = 0;
    for (uint32_t i = 0; i < userModel->nLayers; i++)
    {
        if (userModel->pLayers[i].operation == INTEL_GMM)
        {
            ++gmmCount;
        }
    }

    return gmmCount;
}

BaseValidator CompiledModel::makeValidator()
{
    return BaseValidator
    {
        hwCapabilities,
        ValidBoundariesFunctor {
            [this] (const void *buffer, size_t bufferSize)
            {
                IdentifyBuffer(buffer, bufferSize);
            }
        }
    };
}

uint32_t CompiledModel::scoreAllSubModels(RequestConfiguration& config,
    RequestProfiler *profiler, KernelBuffers *buffers)
{
    const auto& deviceSubmodels = getSubmodels(
            HardwareCapabilities(config.GetConsistentDevice()));
    auto saturationCount = uint32_t{0};
    for (const auto& submodel : deviceSubmodels)
    {
        uint32_t layerIndex = submodel->LayerIndex;
        uint32_t layerCount = submodel->GetLayerCount();
        switch (submodel->Type)
        {
        case Software:
        saturationCount += softwareModel.Score(layerIndex, layerCount, config, profiler, buffers);
        break;
        case Hardware:
        saturationCount += hardwareModel->Score(layerIndex, layerCount, config, profiler, buffers);
        break;
        // TODO: 3: HardwareModel should identify if device is a GMM device
        case GMMHardware:
        saturationCount += hardwareModel->Score(layerIndex, 1, config, profiler, buffers);
        break;
        }
    }
    return saturationCount;
}

const std::vector<std::unique_ptr<SubModel>>&
CompiledModel::getSubmodels(const HardwareCapabilities& hwCaps)
{
    if (submodels.find(hwCaps.GetDeviceVersion()) == submodels.end())
    {
        createSubmodels(hwCaps);
    }

    return submodels.at(hwCaps.GetDeviceVersion());
}

SubmodelType CompiledModel::getSubmodelType(
        const HardwareCapabilities &hwCaps, const Layer& layer) const
{
    auto deviceGeneration = hwCaps.GetDeviceGeneration();
    auto dataConfig = layer.GetDataMode();
    auto supportMapIterator = DataConfig::Capabilities.find(dataConfig);
    if (supportMapIterator == DataConfig::Capabilities.end())
    {
        return SubmodelType::Software;
    }

    const auto& supportMap = supportMapIterator->second;
    auto supportIterator = supportMap.find(layer.Operation);
    if (supportIterator == supportMap.end())
    {
        return SubmodelType::Software;
    }

    const auto& hwSupportMap = supportIterator->second.Hw;
    auto hwSupportIterator = hwSupportMap.find(deviceGeneration);
    if (hwSupportIterator == hwSupportMap.end())
    {
        return SubmodelType::Software;
    }

    auto isSupportedByHardware = hwSupportIterator->second;
    if (!isSupportedByHardware)
    {
        return SubmodelType::Software;
    }

    if (hwCaps.IsLayerSupported(layer.Operation))
    {
        return SubmodelType::Hardware;
    }

    if (INTEL_GMM == layer.Operation && hwCaps.HasFeature(LegacyGMM))
    {
        return SubmodelType::GMMHardware;
    }

    return SubmodelType::Software;
};


void CompiledModel::createSubmodels(const HardwareCapabilities& hwCaps)
{
    auto deviceVersion = hwCaps.GetDeviceVersion();
    auto& deviceSubmodels = submodels[deviceVersion];

    auto layerIndex = uint32_t { 0 };
    auto &layers = softwareModel.Layers;

    auto submodelType = getSubmodelType(hwCaps, *layers.at(layerIndex));

    deviceSubmodels.emplace_back(std::make_unique<SubModel>(submodelType, 0));
    auto currentSubmodel = deviceSubmodels.back().get();
    layerIndex++;

    for (; layerIndex < LayerCount; ++layerIndex)
    {
        const auto& layer = *layers.at(layerIndex);
        submodelType = getSubmodelType(hwCaps, layer);

        auto doSplit = false;

        if (GMMHardware == submodelType
                || currentSubmodel->Type != submodelType)
        {
            doSplit = true;
        }
        else if (Hardware == submodelType
                && currentSubmodel->GetLayerCount() == hwCaps.GetMaximumLayerCount())
        {
            doSplit = true;
        }

        if (doSplit)
        {
            deviceSubmodels.emplace_back(
                    std::make_unique<SubModel>(submodelType, layerIndex));
            currentSubmodel = deviceSubmodels.back().get();
        }
        else
        {
            currentSubmodel->AddLayer();
        }
    }
}
