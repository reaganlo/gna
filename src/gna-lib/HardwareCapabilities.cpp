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

#include <algorithm>

using namespace GNA;

// GNA hardware supports 256MB models, consisting of:
// - layer descriptors
// - user data
const uint32_t HardwareCapabilities::MaximumModelSize = 256 * 1024 * 1024;

std::map<gna_device_version, const GenerationCapabilities> HardwareCapabilities::gnaCapsMap = {
    { GNA_KBL,
        {GMM_DEVICE,
        1,
        {
            { BaseFunctionality, false},
            { CNN, false },
            { LegacyGMM, true },
            { GMMLayer, false },
            { MultiBias, false },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, false },
            { CNN2D, false }
        },
        6,
        {{1, 8}},
        4,
        0,
        0,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_SKL,
        {GMM_DEVICE,
        1,
        {
            { BaseFunctionality, false},
            { CNN, false },
            { LegacyGMM, true },
            { GMMLayer, false },
            { MultiBias, false },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, false },
            { CNN2D, false }
        },
        6,
        {{1, 8}},
        4,
        0,
        0,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_CNL,
        {GNA_0_9,
        1023,
        {
            { BaseFunctionality, true},
            { CNN, false },
            { LegacyGMM, true },
            { GMMLayer, false },
            { MultiBias, false },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, false },
            { CNN2D, false }
        },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_GLK,
        {GNA_1_0,
        1023,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, false },
            { MultiBias, false },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, false },
            { CNN2D, false }
        },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_ICL,
        {GNA_1_0,
        1023,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, false },
            { MultiBias, false },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, false },
            { CNN2D, false }
        },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_SUE_CREEK,
        {GNA_1_0_EMBEDDED,
        1023,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, false },
            { MultiBias, false },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, false },
            { CNN2D, false }
        },
        3,
        {{2, 8}},
        4,
        1,
        1,
        {6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144},},
    },
    { GNA_TGL,
        {GNA_2_0,
        4096,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, true },
            { MultiBias, true },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, true },
            { CNN2D, false }
        },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_JELLYFISH,
        {GNA_2_1_EMBEDDED,
        4096,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, true },
            { MultiBias, true },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, true },
            { CNN2D, false }
        },
        3,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_ADL,
        {GNA_3_0,
        8191,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, true },
            { MultiBias, true },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, true },
            { CNN2D, true }
        },
        8,
        {{1, 16}, {2, 8}},
        4,
        2,
        16,
        {},},
    },
    { GNA_ACE_EMBEDDED,
        {GNA_3_0_EMBEDDED,
        8191,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, true },
            { MultiBias, true },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, true },
            { CNN2D, true }
        },
        8,
        {{1, 16}, {2, 8}},
        4,
        2,
        16,
        {},},
    },
    { GNA_ACE_ANNA,
        {GNA_3_1_AUTONOMUS,
        8191,
        {
            { BaseFunctionality, true},
            { CNN, true },
            { LegacyGMM, true },
            { GMMLayer, true },
            { MultiBias, true },
            { L1Distance, false },
            { L2Distance, false },
            { ComputerVision, false },
            { NewPerformanceCounters, true },
            { CNN2D, true }
        },
        2,
        {{1, 16}, {2, 8}},
        4,
        2,
        16,
        {},},
    },
};

gna_device_version HardwareCapabilities::GetDeviceVersion(gna_device_generation generation)
{
    auto type = std::find_if(gnaCapsMap.cbegin(), gnaCapsMap.cend(),
        [generation](const std::pair<const gna_device_version, const GenerationCapabilities>& genCaps)
            {
                return genCaps.second.Generation == generation;
            });
    return type->first;
}

uint32_t HardwareCapabilities::GetMaximumLayerCount(gna_device_version hwId)
{
    return gnaCapsMap.at(hwId).MaximumLayerCount;
}

uint32_t HardwareCapabilities::GetComputeEngineCount(gna_device_version hwId)
{
    return gnaCapsMap.at(hwId).ComputeEngineCount;
}

uint32_t HardwareCapabilities::GetBufferSizeInKB(gna_device_version hwId)
{
    auto caps = gnaCapsMap.at(hwId);
    return caps.ComputeEngineCount * caps.BufferSizesPerCEInKB;
}

uint32_t HardwareCapabilities::GetBufferSizeInKB() const
{
    auto caps = gnaCapsMap.at(deviceVersion);
    return caps.ComputeEngineCount * caps.BufferSizesPerCEInKB;
}

uint32_t HardwareCapabilities::GetBufferElementCount(
    gna_device_version hwId, uint32_t grouping, uint32_t inputPrecision)
{
    if (hwId == GNA_ADL || hwId == GNA_ACE_EMBEDDED || hwId == GNA_ACE_ANNA)
    {
        const auto ceCount = gnaCapsMap.at(hwId).ComputeEngineCount;
        auto count = (GetBufferSizeInKB(hwId) * 1024)
            / (ceCount *  16 * grouping);
        count *= ceCount * 16 / inputPrecision;
        count *= grouping;

        return count;
    }
    else
    {
        return gnaCapsMap.at(hwId).BufferElementCountBackward[grouping - 1];
    }
}

HardwareCapabilities::HardwareCapabilities(
    gna_device_version deviceVersionIn) :
    deviceVersion{deviceVersionIn},
    bufferSize {GetBufferSizeInKB()}
{
}

void HardwareCapabilities::DiscoverHardware(DriverInterface &driverInterface)
{
    if (!driverInterface.IsDeviceOpened())
    {
        return;
    }

    try
    {
        auto driverCapabilities = driverInterface.GetCapabilities();

        //TODO:3: remove when ADL bug with input buffer will be fixed
        if (driverCapabilities.hwId == GNA_ADL)
            driverCapabilities.hwInBuffSize = 32;

        Expect::Equal((size_t)1, gnaCapsMap.count(driverCapabilities.hwId), GNA_DEVNOTFOUND);
        Expect::Equal(driverCapabilities.hwInBuffSize, GetBufferSizeInKB(driverCapabilities.hwId),
            status_t::GNA_ERR_INVALID_DEVICE_VERSION);

        deviceVersion = driverCapabilities.hwId;
        bufferSize = driverCapabilities.hwInBuffSize;
        driverRecoveryTimeout = driverCapabilities.recoveryTimeout;

        hardwareSupported = true;
    }
    catch (GnaException&)
    {
        Log->Message("No compatible hardware detected.\n");
    }
}

void HardwareCapabilities::GetHardwareConsistencySettings(uint32_t bufferElementCount[2 * XNN_N_GROUP_MAX],
    gna_device_version hwId)
{
    for (uint32_t p = 0; p < 2; p++)
    {
        for (uint32_t i = 0; i < XNN_N_GROUP_MAX; i++)
        {
            bufferElementCount[p * XNN_N_GROUP_MAX +  i] = GetBufferElementCount(hwId, i + 1, p + 1);
        }
    }
}

gna_device_version HardwareCapabilities::GetDeviceVersion() const
{
    return deviceVersion;
}

gna_device_generation HardwareCapabilities::GetDeviceGeneration() const
{
    return gnaCapsMap.at(deviceVersion).Generation;
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
    if (!hardwareSupported)
        return false;

    const auto& caps = gnaCapsMap.at(deviceVersion);
    return caps.Features.at(feature);
}

uint32_t HardwareCapabilities::GetMaximumLayerCount() const
{
    return GetMaximumLayerCount(deviceVersion);
}
