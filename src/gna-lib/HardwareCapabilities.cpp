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

#include "HardwareCapabilities.h"

#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"
#include "Macros.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

using namespace GNA;

// GNA hardware supports 256MB models, consisting of:
// - layer descriptors
// - user data
const uint32_t HardwareCapabilities::MaximumModelSize = 256 * 1024 * 1024;

#define EVALUATOR(x)  #x

template<Gna2DeviceVersion version>
static GenerationCapabilities GetVerCaps();

template<Gna2DeviceVersion baseVersion, Gna2DeviceGeneration targetGeneration>
static GenerationCapabilities DeriveVerCaps()
{
    static GenerationCapabilities caps = GetVerCaps<baseVersion>();
    caps.Generation = targetGeneration;
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionGMM>()
{
    return { Gna2DeviceGenerationGmm,
            1,
            {
                { BaseFunctionality, false},
                { CNN, false },
                { LegacyGMM, true },
                { GMMLayer, false },
                { MultiBias, false },
                { NewPerformanceCounters, false },
                { CNN2D, false }
            },
            6,
            {{1, 8}},
            4,
            0,
            0,
            {0, 0, 0, 0, 0, 0, 0, 0,
                12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
            {},
            {}
    };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion0_9>()
{
    return {
    Gna2DeviceGeneration0_9,
    1023,
    {
        { BaseFunctionality, true},
        { CNN, false },
        { LegacyGMM, true },
        { GMMLayer, false },
        { MultiBias, false },
        { NewPerformanceCounters, false },
        { CNN2D, false }
    },
    6,
    {{2, 8}},
    4,
    1,
    1,
    {0, 0, 0, 0, 0, 0, 0, 0,
    12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
    {},
    {}
    };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion1_0>()
{
    static auto caps = DeriveVerCaps<Gna2DeviceVersion0_9, Gna2DeviceGeneration1_0>();
    caps.Features[CNN] = true;
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionEmbedded1_0>()
{
    static auto caps = DeriveVerCaps<Gna2DeviceVersion1_0, Gna2DeviceGeneration1_0>();
    caps.ComputeEngineCount = 3;
    caps.BufferElementCount =
    { 0, 0, 0, 0, 0, 0, 0, 0,
        6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144 };
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion2_0>()
{
    return  { Gna2DeviceGeneration2_0,
           4096,
           {
               { BaseFunctionality, true},
               { CNN, true },
               { LegacyGMM, true },
               { GMMLayer, true },
               { MultiBias, true },
               { NewPerformanceCounters, true },
               { CNN2D, false }
           },
           6,
           {{2, 8}},
           4,
           1,
           1,
           {0, 0, 0, 0, 0, 0, 0, 0,
               12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
           {},
            {} };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion3_0>()
{
    return { Gna2DeviceGeneration3_0,
                8192,
                {
                    { BaseFunctionality, true},
                    { CNN, true },
                    { LegacyGMM, true },
                    { GMMLayer, true },
                    { MultiBias, true },
                    { NewPerformanceCounters, true },
                    { CNN2D, true }
                },
                8,
                {{1, 16}, {2, 8}},
                4,
                2,
                16,
                {},
                {},
                EVALUATOR(GNA_HW_MODULE_30)};
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionEmbedded3_1>()
{
    return DeriveVerCaps<Gna2DeviceVersion3_0, Gna2DeviceGeneration3_1>();
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion3_5>()
{
    static auto caps = DeriveVerCaps<Gna2DeviceVersion3_0, Gna2DeviceGeneration3_5>();
    caps.HwModuleName = EVALUATOR(GNA_HW_MODULE_35);
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionEmbedded3_5>()
{
    return DeriveVerCaps<Gna2DeviceVersion3_5, Gna2DeviceGeneration3_5>();
}

template<Gna2DeviceVersion version>
static DevVerGenMap::allocator_type::value_type GetCaps()
{
    return { version, GetVerCaps<version>() };
}

DevVerGenMap& HardwareCapabilities::getCapsMap()
{
    static DevVerGenMap capsMap = {
         GetCaps<Gna2DeviceVersionGMM>(),
         GetCaps<Gna2DeviceVersion0_9>(),
         GetCaps<Gna2DeviceVersion1_0>(),
         GetCaps<Gna2DeviceVersionEmbedded1_0>(),
         GetCaps<Gna2DeviceVersion2_0>(),
         GetCaps<Gna2DeviceVersion3_0>(),
         GetCaps<Gna2DeviceVersion3_5>(),
         GetCaps<Gna2DeviceVersionEmbedded3_1>(),
         GetCaps<Gna2DeviceVersionEmbedded3_5>(),
    };

    // initialize remaining items that depend on capsMap values
    if (capsMap.at(Gna2DeviceVersion3_0).BufferElementCount[0] == 0)
    {
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersion3_0)));
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersion3_0)), true);
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersion3_5)));
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersion3_5)), true);
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionEmbedded3_1)));
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionEmbedded3_1)), true);
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionEmbedded3_5)));
        initHardwareConsistencySettingsAdl(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionEmbedded3_5)), true);
    }

    return capsMap;
}

const GenerationCapabilities&
HardwareCapabilities::getGenerationCapabilities(DeviceVersion deviceVersionIn)
{
    try
    {
        return getCapsMap().at(deviceVersionIn);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }
}

bool HardwareCapabilities::IsAdlGeneration(Gna2DeviceGeneration generation)
{
    return Gna2DeviceGeneration3_0 == generation || Gna2DeviceGeneration3_5 == generation;
}

bool HardwareCapabilities::IsAdlDevice(DeviceVersion deviceVersionIn)
{
    auto const caps = getGenerationCapabilities(deviceVersionIn);
    return IsAdlGeneration(caps.Generation);
}

DeviceVersion HardwareCapabilities::GetDeviceVersion(Gna2DeviceGeneration generation)
{
    auto type = std::find_if(getCapsMap().cbegin(), getCapsMap().cend(),
        [generation](const std::pair<const DeviceVersion, const GenerationCapabilities>& genCaps)
    {
        return genCaps.second.Generation == generation;
    });
    return type->first;
}

uint32_t HardwareCapabilities::GetMaximumLayerCount(DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).MaximumLayerCount;
}

uint32_t HardwareCapabilities::GetComputeEngineCount(DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).ComputeEngineCount;
}

uint32_t HardwareCapabilities::GetBufferElementCount(
    DeviceVersion deviceVersionIn, uint32_t grouping, uint32_t inputPrecision)
{
    auto const index = ((inputPrecision - 1) * BufferArraySizeSingle) + grouping - 1;
    return getGenerationCapabilities(deviceVersionIn).BufferElementCount[index];
}

HardwareCapabilities::HardwareCapabilities(
    DeviceVersion deviceVersionIn) :
    deviceVersion{deviceVersionIn},
    bufferSize{GetBufferSizeInKB()}
{
}

void HardwareCapabilities::DiscoverHardware(const DriverCapabilities& discoveredDriver)
{
    //TODO:3: remove when ADL bug with input buffer will be fixed
    auto hwInBuffSize = discoveredDriver.hwInBuffSize;
    if (discoveredDriver.deviceVersion == Gna2DeviceVersion3_0 || discoveredDriver.deviceVersion == Gna2DeviceVersion3_5)
    {
        hwInBuffSize = 32;
    }
    if (1 != getCapsMap().count(discoveredDriver.deviceVersion) ||
        hwInBuffSize != GetBufferSizeInKB(discoveredDriver.deviceVersion))
    {
        Log->Message("No compatible hardware detected.\n");
        return;
    }

    deviceVersion = discoveredDriver.deviceVersion;
    bufferSize = hwInBuffSize;

    hardwareSupported = true;
}

void HardwareCapabilities::OverrideDeviceVersion(Gna2DeviceVersion deviceOverride)
{
    overridenDeviceVersion = true;
    deviceVersion = deviceOverride;
    bufferSize = GetBufferSizeInKB();
}

uint32_t const * HardwareCapabilities::GetHardwareConsistencySettings(
    DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).BufferElementCount.data();
}

uint32_t const * HardwareCapabilities::GetHardwareConsistencySettingsForAdl(DeviceVersion deviceVersionIn)
{
    if (IsAdlDevice(deviceVersionIn))
    {
        return getGenerationCapabilities(deviceVersionIn).BufferElementCountAdlWorkaround.data();
    }
    return nullptr;
}

DeviceVersion HardwareCapabilities::GetDeviceVersion() const
{
    return deviceVersion;
}

const char* HardwareCapabilities::GetHwModuleName() const
{
    return getGenerationCapabilities(deviceVersion).HwModuleName.c_str();
}

Gna2DeviceGeneration HardwareCapabilities::GetDeviceGeneration() const
{
    return getGenerationCapabilities(deviceVersion).Generation;
}

bool HardwareCapabilities::IsAdlGeneration() const
{
    auto const generation = GetDeviceGeneration();
    return IsAdlGeneration(generation);
}

bool HardwareCapabilities::IsLayerSupported(nn_operation operation) const
{
    static const std::map<nn_operation, GnaFeature> featureMap =
    {
        {INTEL_AFFINE, BaseFunctionality},
        {INTEL_AFFINE_DIAGONAL, BaseFunctionality},
        {INTEL_COPY, BaseFunctionality},
        {INTEL_DEINTERLEAVE, BaseFunctionality},
        {INTEL_INTERLEAVE, BaseFunctionality},
        {INTEL_RECURRENT, BaseFunctionality},
        {INTEL_AFFINE_MULTIBIAS, MultiBias},
        {INTEL_CONVOLUTIONAL, CNN},
        {INTEL_GMM, GMMLayer},
        {INTEL_CONVOLUTIONAL_2D, CNN2D},
    };

    return HasFeature(featureMap.at(operation));
}

bool HardwareCapabilities::HasFeature(GnaFeature feature) const
{
    const auto& caps = getGenerationCapabilities(deviceVersion);
    return caps.Features.at(feature);
}

uint32_t HardwareCapabilities::GetMaximumLayerCount() const
{
    return GetMaximumLayerCount(deviceVersion);
}

void HardwareCapabilities::initHardwareConsistencySettingsAdl(GenerationCapabilities& caps, bool isWorkaround)
{
    auto& buffers = (isWorkaround) ? caps.BufferElementCountAdlWorkaround : caps.BufferElementCount;
    for (uint32_t p = 0; p < 2; p++)
    {
        auto const inputPrecision = (isWorkaround) ? 2 : p + 1;
        for (uint32_t i = 0; i < BufferArraySizeSingle; i++)
        {
            buffers[p * BufferArraySizeSingle + i] =
                getBufferElementCountAdl(caps.ComputeEngineCount, caps.BufferSizesPerCEInKB, i + 1, inputPrecision);
        }
    }
}

uint32_t HardwareCapabilities::getBufferElementCountAdl(uint32_t ceCount, uint32_t bufferSizeInKB,
    uint32_t grouping, uint32_t inputPrecision)
{
    auto count = (bufferSizeInKB * 1024)
        / (16 * grouping);
    count *= ceCount * 16 / inputPrecision;
    count *= grouping;

    return count;
}

uint32_t HardwareCapabilities::GetBufferSizeInKB(DeviceVersion deviceVersionIn)
{
    auto const& caps = getGenerationCapabilities(deviceVersionIn);
    return caps.ComputeEngineCount * caps.BufferSizesPerCEInKB;
}

uint32_t HardwareCapabilities::GetBufferSizeInKB() const
{
    return GetBufferSizeInKB(deviceVersion);
}
