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

#define _COMPONENT_ "GmmDispatcher::"

#ifdef _WIN32
#include <intrin.h>
#endif

#include "AccelerationDetector.h"
#include "GnaException.h"

using std::shared_ptr;
using std::make_shared;
using std::array;
using std::map;

using namespace GNA;

/**
 * Masks for CPU extensions detection
 */
#define SSE4_MASK 0x00180000  // mask for SSE4_1, SSE4_2 feature flags, 19,20 bits
#define AVX1_MASK 0x18000000  // mask for OSXSAVE and AVX feature flags, 27,28 bits
#define AVX2_MASK 0x00000020  // mask for AVX2 feature flag, 5 bit
#define XYMM_MASK 0x00000006  // mask for OS enabled XMM+YMM state support flag, 1,2 bits
#define AVX2_MASK_ASM 000000020H  // mask for AVX2 feature flag, 5 bit

/**
 * If _XCR_XFEATURE_ENABLED_MASK is not defined set it to 0
 * intrin.h header file containing this flag is MS-specific
 * and not available on linux platforms
 */
#ifndef _XCR_XFEATURE_ENABLED_MASK
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

const map<GnaDeviceType, array<bool, GnaFeatureCount>>
AccelerationDetector::gnaFeatureMap = {
                        // Basic, CNN,   GMM,  GMMLayer, MultiBias, L1Dist, L2Dist, ComputerVision
        { GNA_DEV_CNL,     {true, false, true, false,    false,     false,  false,  false} },
        { GNA_DEV_GLK,     {true, true,  true, false,    false,     false,  false,  false} },
        { GNA_DEV_LKF,     {true, true,  true, true,     true,      false,  false,  false} },
        { GNA_DEV_TGL,     {true, true,  true, true,     true,      false,  false,  false} },
        { GNA_DEV_UNKNOWN, {true, true,  true, true,     true,      false,  false,  false} },
    };

AccelerationDetector::AccelerationDetector(): fastestAcceleration(GNA_GEN_FAST)
{
    for (auto& acc : accelerationModes)
    {
        acc.second = ACC_NOTSUPPORTED;
    }

    accelerationModes[GNA_GEN_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_GEN_FAST] = ACC_SUPPORTED;
}

void AccelerationDetector::discoverHardwareExistence()
{
    try
    {
        IoctlSender::Open(GUID_DEVINTERFACE_GNA_DRV);
        accelerationModes[GNA_HW] = ACC_SUPPORTED;
    }
    catch (GnaException e)
    {
        accelerationModes[GNA_HW] = ACC_NOTSUPPORTED;
        LOG("Hardware not detected.");
    }
}

void AccelerationDetector::discoverHardwareCapabilities()
{
    if (!IsHardwarePresent()) return;

    IoctlSend(GNA_IOCTL_CPBLTS, nullptr, 0, &deviceCapabilities, sizeof(GNA_CPBLTS));
}

const uint32_t AccelerationDetector::GetHardwareBufferSize() const
{
    if (IsHardwarePresent())
    {
        return deviceCapabilities.hwInBuffSize;
    }
    else
    {
        throw GnaException(GNA_DEVNOTFOUND);
    }
}

bool AccelerationDetector::IsHardwarePresent() const
{
    return ACC_SUPPORTED == accelerationModes.at(GNA_HW);
}

bool AccelerationDetector::IsLayerSupported(intel_layer_kind_t layerType) const
{
    if (!IsHardwarePresent()) return false;
    const auto& deviceFeatureMap = gnaFeatureMap.at(deviceCapabilities.device_type);
    switch(layerType)
    {
        case INTEL_AFFINE:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_AFFINE_DIAGONAL:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_AFFINE_MULTIBIAS:
            return deviceFeatureMap[MultiBias];
        case INTEL_CONVOLUTIONAL:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_COPY:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_DEINTERLEAVE:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_INTERLEAVE:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_RECURRENT:
            return deviceFeatureMap[BaseFunctionality];
        case INTEL_GMM:
            return deviceFeatureMap[GMMLayer];
        default:
            return false;
    }

}

void AccelerationDetector::DetectAccelerations()
{
    discoverHardwareExistence();
    discoverHardwareCapabilities();

    int cpuId[4];           // cpu id string
    int sse4 = ACC_NOTSUPPORTED; // is Intel SSE4 Extensions support available
    int avx1 = ACC_NOTSUPPORTED; // Intel ACX1 Extensions support available
    int avx2 = ACC_NOTSUPPORTED; // Intel ACX2 Extensions support available
    unsigned long long xcrFeature = 0;

    // get CPU IDs
    __cpuid(cpuId, 1);

    // detect SSE4
    // check both SSE4_1, SSE4_2 feature flags (bits 19,20)
    sse4 = cpuId[2] & SSE4_MASK;
    if (SSE4_MASK == sse4)
    {
        accelerationModes[GNA_SSE4_2_FAST] = ACC_SUPPORTED;
        accelerationModes[GNA_SSE4_2_SAT]  = ACC_SUPPORTED;
        fastestAcceleration = GNA_SSE4_2_FAST;
    }

    // detect AVX1 & AVX2
    // check OSXSAVE and AVX feature flags, bits 27,28
    avx1 = cpuId[2] & AVX1_MASK;
    if (AVX1_MASK == avx1) // processor supports AVX instructions and XGETBV is enabled by OS
    {
        // AVX2 requires AVX1 support
        // get AVX2 flag, bit 5
        //avx2 = cpuId2[1] & AVX2_MASK;
        // below asm code is equivalent to line above
        _asm
        {
            mov eax, 7
                mov ecx, 0
                cpuid; get CPU ID
                and ebx, AVX2_MASK_ASM; compare CPU ID and AVX2 mask
                mov avx2, ebx; save result in avx2
        }

        // check OS has enabled both XMM and YMM state support
        xcrFeature = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        xcrFeature = xcrFeature & XYMM_MASK;
        if (XYMM_MASK == xcrFeature)
        {
            accelerationModes[GNA_AVX1_FAST] = ACC_SUPPORTED;
            accelerationModes[GNA_AVX1_SAT]  = ACC_SUPPORTED;
            fastestAcceleration = GNA_AVX1_FAST;

            // check AVX2 flag, bit 5
            if (AVX2_MASK & avx2)
            {
                accelerationModes[GNA_AVX2_FAST] = ACC_SUPPORTED;
                accelerationModes[GNA_AVX2_SAT]  = ACC_SUPPORTED;
                fastestAcceleration = GNA_AVX2_FAST;
            }
        }
    }

    LOG("GNA_HW        %d\n", accelerationModes[GNA_HW]);
    LOG("GNA_AUTO_FAST %d\n", accelerationModes[GNA_AUTO_FAST]);
    LOG("GNA_AUTO_SAT  %d\n", accelerationModes[GNA_AUTO_SAT]);
    LOG("GNA_AVX2_FAST %d\n", accelerationModes[GNA_AVX2_FAST]);
    LOG("GNA_AVX2_SAT  %d\n", accelerationModes[GNA_AVX2_SAT]);
    LOG("GNA_AVX1_FAST %d\n", accelerationModes[GNA_AVX1_FAST]);
    LOG("GNA_AVX1_SAT  %d\n", accelerationModes[GNA_AVX1_SAT]);
    LOG("GNA_SSE4_FAST %d\n", accelerationModes[GNA_SSE4_2_FAST]);
    LOG("GNA_SSE4_SAT  %d\n", accelerationModes[GNA_SSE4_2_SAT]);
    LOG("GNA_GEN_FAST  %d\n", accelerationModes[GNA_GEN_FAST]);
    LOG("GNA_GEN_SAT   %d\n", accelerationModes[GNA_GEN_SAT]);
}

acceleration AccelerationDetector::GetFastestAcceleration() const
{
    return fastestAcceleration;
}