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

#pragma once

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <array>
#include <cstdint>
#include <map>

#include "gna2-common-impl.h"

namespace GNA
{
class DriverInterface;

enum GnaFeature
{
    BaseFunctionality = 0, // DNN, DNN_AL, DIAGONAL, RNN, COPY, TRANSPOSE, PWL
    CNN,
    LegacyGMM,
    GMMLayer,
    MultiBias,
    L1Distance,
    L2Distance,
    ComputerVision,
    NewPerformanceCounters,
    CNN2D,
};

// buffer array size for single precision
static constexpr uint32_t BufferArraySizeSingle = XNN_N_GROUP_MAX;
static constexpr uint32_t BufferArraySize = 2 * BufferArraySizeSingle;

struct GenerationCapabilities
{
    gna_device_generation Generation;
    uint32_t MaximumLayerCount;
    std::map<GnaFeature, bool> Features;
    uint32_t ComputeEngineCount;
    std::map<const uint32_t /* input precision */, const uint32_t> MacCountPerCE;
    uint32_t BufferSizesPerCEInKB;
    uint32_t PoolingEngineCountPerCE;
    uint32_t ActivationEngineCount;
    std::array<uint32_t, BufferArraySize> BufferElementCount;
    std::array<uint32_t, BufferArraySize> BufferElementCountAdlWorkaround;
};

class HardwareCapabilities
{
public:
    explicit HardwareCapabilities(DeviceVersion deviceVersionIn = DefaultDeviceVersion);

    void DiscoverHardware(DriverInterface &driverInterface);

    static uint32_t const * GetHardwareConsistencySettings(DeviceVersion hwId);
    static uint32_t const * GetHardwareConsistencySettingsForAdl(DeviceVersion hwId);

    // For now all hardware generations share the same maximum model size
    // in the future it's possible to integrate it as GenerationCapabilities field
    static const uint32_t MaximumModelSize;

    static bool IsAdlGeneration(gna_device_generation generation);
    static bool IsAdlDevice(DeviceVersion deviceVersion);

    static DeviceVersion GetDeviceVersion(gna_device_generation generation);

    static uint32_t GetMaximumLayerCount(DeviceVersion hwId);

    static uint32_t GetComputeEngineCount(DeviceVersion hwId);

    // Gets the number of data elements that may be stored in hw buffer
    static uint32_t GetBufferElementCount(DeviceVersion hwId,
        uint32_t grouping, uint32_t inputPrecision = GNA_INT16);

    uint32_t GetBufferElementCount(uint32_t grouping, uint32_t inputPrecision = GNA_INT16) const
    {
        return GetBufferElementCount(deviceVersion, grouping, inputPrecision);
    }

    DeviceVersion GetDeviceVersion() const;

    DeviceVersion GetHardwareDeviceVersion() const
    {
        return IsHardwareSupported()
            ? GetDeviceVersion()
            : Gna2DeviceVersionSoftwareEmulation;
    }

    gna_device_generation GetDeviceGeneration() const;

    bool IsAdlGeneration() const;

    uint32_t GetMaximumLayerCount() const;

    bool IsLayerSupported(nn_operation operation) const;

    bool IsHardwareSupported() const
    {
        return hardwareSupported;
    }

    bool HasFeature(GnaFeature feature) const;

private:
    static std::map<DeviceVersion, const GenerationCapabilities>& getCapsMap();

    static const GenerationCapabilities& getGenerationCapabilities(DeviceVersion deviceVersionIn);

    static void initHardwareConsistencySettingsAdl(GenerationCapabilities& caps, bool isWorkaround = false);

    static uint32_t getBufferElementCountAdl(uint32_t ceCount, uint32_t bufferSizeInKB,
        uint32_t grouping, uint32_t inputPrecision = GNA_INT16);

    uint32_t GetBufferSizeInKB() const;

    static uint32_t GetBufferSizeInKB(DeviceVersion hwId);

    bool hardwareSupported = false;

    DeviceVersion deviceVersion;

    uint32_t bufferSize;

    uint32_t driverRecoveryTimeout = 0;
};

}
