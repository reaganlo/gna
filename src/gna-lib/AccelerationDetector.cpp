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

#if defined(__GNUC__)
#include <cpuid.h>
static inline unsigned long long _xgetbv(unsigned int ctr)
{
    int a, d;
    __asm("xgetbv" : "=a"(a),"=d"(d) : "c"(ctr) : );
    return a | (((unsigned long long) d) << 32);
}
#define cpuid(info, level) __cpuid_count(level, 0, info[0], info[1], info[2], info[3])
#else
#if !defined(_MSC_VER)
#include <immintrin.h>
#elif defined(__INTEL_COMPILER)
#include <intrin.h>
#endif // __INTEL_COMPILER
#define cpuid(info, level) __cpuidex((int*)(info), level, 0)
#endif // __GNUC__

#include "AccelerationDetector.h"
#include "GnaException.h"
#include "Logger.h"

#include <map>
#include <array>

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

const std::map<gna_device_kind, std::array<bool, GnaFeatureCount>>
AccelerationDetector::gnaFeatureMap = {
    // Basic, CNN,   LegacyGMM,  GMMLayer, MultiBias, L1Dist, L2Dist, ComputerVision, Layer8K, NewPerformanceCounters
{ GNA_CNL,   {true, false, true,  false,    false,     false,  false,  false, false, false } },
{ GNA_GLK,   {true, true,  true,  false,    false,     false,  false,  false, false, false } },
{ GNA_ICL,   {true, true,  true,  false,    false,     false,  false,  false, false, false } },
{ GNA_TGL,   {true, true,  false, true,     true,      false,  false,  false, true,  true  } },
{ GNA_SUE,   {true, true,  true,  false,    false,     false,  false,  false, false, false } },
{ GNA_SUE_2, {true, true,  false, true,     true,      false,  false,  false, true,  true  } }
};

std::map<const WeightMode, std::map<const acceleration, const AffineKernel>> AccelerationDetector::AffineKernels = {
    {GNA_WEIGHT_1B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.affineSingle1Bfull },
        { GNA_GEN_FAST,    xnnKernel_generic.affineSingle1Bfull }}
    },
    {GNA_WEIGHT_2B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.affineSingle2Bfull },
        { GNA_GEN_FAST,    xnnKernel_generic.affineSingle2Bfull }}
    }
};

std::map<const WeightMode, std::map<const acceleration, const AffineActiveListKernel>> AccelerationDetector::AffineKernelsAl = {
    {GNA_WEIGHT_1B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.affineSingle1Bal },
        { GNA_GEN_FAST,    xnnKernel_generic.affineSingle1Bal }}
    },
    {GNA_WEIGHT_2B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.affineSingle2Bal },
        { GNA_GEN_FAST,    xnnKernel_generic.affineSingle2Bal }}
    }
};

std::map<const WeightMode, std::map<const acceleration, const AffineKernel>> AccelerationDetector::MultibiasKernels = {
    {GNA_WEIGHT_1B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.affineMulti1B },
        { GNA_GEN_FAST,    xnnKernel_generic.affineMulti1B }}
    },
    {GNA_WEIGHT_2B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.affineMulti2B },
        { GNA_GEN_FAST,    xnnKernel_generic.affineMulti2B }}
    }
};

std::map<const WeightMode, std::map<const acceleration, const RecurrentKernel>> AccelerationDetector::RecurrentKernels = {
    {GNA_WEIGHT_1B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.recurrent1B },
        { GNA_GEN_FAST,    xnnKernel_generic.recurrent1B }}
    },
    {GNA_WEIGHT_2B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.recurrent2B },
        { GNA_GEN_FAST,    xnnKernel_generic.recurrent2B }}
    }
};

std::map<const WeightMode, std::map<const acceleration, const AffineKernel>> AccelerationDetector::DiagonalKernels = {
    {GNA_WEIGHT_1B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.diagonal1B },
        { GNA_GEN_FAST,    xnnKernel_generic.diagonal1B }}
    },
    {GNA_WEIGHT_2B,
       {{ GNA_GEN_SAT,     xnnKernel_generic_sat.diagonal2B },
        { GNA_GEN_FAST,    xnnKernel_generic.diagonal2B }}
    }
};

std::map<const acceleration, const TransposeKernel> AccelerationDetector::TransposeKernels = {
    { GNA_GEN_SAT,     xnnKernel_generic_sat.transpose },
    { GNA_GEN_FAST,    xnnKernel_generic.transpose }
};

std::map<const acceleration, const CopyKernel> AccelerationDetector::CopyKernels = {
    { GNA_GEN_SAT,     xnnKernel_generic_sat.copy },
    { GNA_GEN_FAST,    xnnKernel_generic.copy }
};

std::map<const acceleration, const ConvolutionKernel> AccelerationDetector::ConvolutionKernels = {
    { GNA_GEN_SAT,     xnnKernel_generic_sat.convolution },
    { GNA_GEN_FAST,    xnnKernel_generic.convolution }
};

std::map<const acceleration, const ConvolutionPoolingKernel> AccelerationDetector::PoolingKernels = {
    { GNA_GEN_SAT,     xnnKernel_generic_sat.convolutionPooling },
    { GNA_GEN_FAST,    xnnKernel_generic.convolutionPooling }
};

std::map<const acceleration, const PwlKernel> AccelerationDetector::PwlKernels = {
    { GNA_GEN_SAT,     xnnKernel_generic_sat.pwl },
    { GNA_GEN_FAST,    xnnKernel_generic.pwl }
};

std::map<const gna_gmm_mode, std::map<const acceleration, const GmmMaxMix>> AccelerationDetector::GmmKernels = {
    { GNA_MAXMIX8, {{ GNA_GEN_SAT,  gmmKernel_generic.gmmMaxMix8 },
                 { GNA_GEN_FAST,    gmmKernel_generic.gmmMaxMix8 }}},

    { GNA_MAXMIX16, {{ GNA_GEN_SAT,  gmmKernel_generic.gmmMaxMix16 },
                 { GNA_GEN_FAST,     gmmKernel_generic.gmmMaxMix16 }}}
};

std::map<const gna_gmm_mode, std::map<const acceleration, const GmmMaxMixActiveList>> AccelerationDetector::GmmActiveListKernels = {
    { GNA_MAXMIX8, {{ GNA_GEN_SAT,  gmmKernel_generic.gmmMaxMix8ActiveList },
                 { GNA_GEN_FAST,    gmmKernel_generic.gmmMaxMix8ActiveList }}},

    { GNA_MAXMIX16, {{ GNA_GEN_SAT,  gmmKernel_generic.gmmMaxMix16ActiveList },
                 { GNA_GEN_FAST,     gmmKernel_generic.gmmMaxMix16ActiveList }}}
};

std::map<acceleration const, std::string const> AccelerationDetector::accelerationNames
{
    {GNA_HW, "GNA_HW"},
    {GNA_AUTO_SAT, "GNA_AUTO_SAT"},
    {GNA_AUTO_FAST, "GNA_AUTO_FAST"},
    {GNA_SW_SAT, "GNA_SW_SAT"},
    {GNA_SW_FAST, "GNA_SW_FAST"},
    {GNA_GEN_SAT, "GNA_GEN_SAT"},
    {GNA_GEN_FAST, "GNA_GEN_FAST"},
    {GNA_SSE4_2_SAT, "GNA_SSE4_2_SAT"},
    {GNA_SSE4_2_FAST, "GNA_SSE4_2_FAST"},
    {GNA_AVX1_SAT, "GNA_AVX1_SAT"},
    {GNA_AVX1_FAST, "GNA_AVX1_FAST"},
    {GNA_AVX2_SAT, "GNA_AVX2_SAT"},
    {GNA_AVX2_FAST, "GNA_AVX2_FAST"},
    {NUM_GNA_ACCEL_MODES, "UNKNOWN ACCELERATION"}
};

void AccelerationDetector::discoverHardware()
{
    accelerationModes[GNA_HW] = ACC_NOTSUPPORTED;
    try
    {
        ioctlSender.Open();
        deviceCapabilities = ioctlSender.GetDeviceCapabilities();
        accelerationModes[GNA_HW] = ACC_SUPPORTED;
    }
    catch (GnaException e)
    {
        accelerationModes[GNA_HW] = ACC_NOTSUPPORTED;
        Log->Message("No compatible hardware detected.\n");
    }
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
    const auto& deviceFeatureMap = gnaFeatureMap.at(deviceCapabilities.deviceKind);
    switch (layerType)
    {
    case INTEL_AFFINE:
        return deviceFeatureMap[BaseFunctionality];
    case INTEL_AFFINE_DIAGONAL:
        return deviceFeatureMap[BaseFunctionality];
    case INTEL_AFFINE_MULTIBIAS:
        return deviceFeatureMap[MultiBias];
    case INTEL_CONVOLUTIONAL:
        return deviceFeatureMap[CNN];
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

bool AccelerationDetector::HasFeature(GnaFeature feature) const
{
    if (!IsHardwarePresent()) return false;

    const auto& deviceFeatureMap = gnaFeatureMap.at(deviceCapabilities.deviceKind);
    return deviceFeatureMap.at(feature);
}

AccelerationDetector::AccelerationDetector(IoctlSender &senderIn) :
    fastestAcceleration{ GNA_GEN_FAST },
    ioctlSender{ senderIn }
{
    // generic, fastest software and auto always supported
    accelerationModes[GNA_GEN_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_GEN_FAST] = ACC_SUPPORTED;
    accelerationModes[GNA_SW_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_SW_FAST] = ACC_SUPPORTED;
    accelerationModes[GNA_AUTO_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_AUTO_FAST] = ACC_SUPPORTED;

    discoverHardware();

    unsigned int cpuId[4];           // cpu id string
    unsigned long long xcrFeature = 0;

    cpuid(cpuId, 0);
    int largestFunctionId = cpuId[0];

    // get CPU IDs
    cpuid(cpuId, 1);

    // detect SSE4
    // check both SSE4_1, SSE4_2 feature flags (bits 19,20)
    if ((cpuId[2] & SSE4_MASK) == SSE4_MASK)
    {
        accelerationModes[GNA_SSE4_2_FAST] = ACC_SUPPORTED;
        accelerationModes[GNA_SSE4_2_SAT] = ACC_SUPPORTED;
        fastestAcceleration = GNA_SSE4_2_FAST;
    }

    if ((cpuId[2] & AVX1_MASK) == AVX1_MASK)
    {
        // check OS has enabled both XMM and YMM state support
        xcrFeature = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        xcrFeature = xcrFeature & XYMM_MASK;
        if (XYMM_MASK == xcrFeature)
        {
            accelerationModes[GNA_AVX1_FAST] = ACC_SUPPORTED;
            accelerationModes[GNA_AVX1_SAT] = ACC_SUPPORTED;
            fastestAcceleration = GNA_AVX1_FAST;
        }

        // check AVX2 flag
        if (largestFunctionId >= 7)
        {
            cpuid(cpuId, 7);
            if ((cpuId[1] & AVX2_MASK) == AVX2_MASK)
            {
                accelerationModes[GNA_AVX2_FAST] = ACC_SUPPORTED;
                accelerationModes[GNA_AVX2_SAT] = ACC_SUPPORTED;
                fastestAcceleration = GNA_AVX2_FAST;
            }
        }
    }

    Log->Message("%s\t%d\n", accelerationNames[GNA_HW].c_str(), accelerationModes[GNA_HW]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_AUTO_FAST].c_str(), accelerationModes[GNA_AUTO_FAST]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_AUTO_SAT].c_str(), accelerationModes[GNA_AUTO_SAT]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_AVX2_FAST].c_str(), accelerationModes[GNA_AVX2_FAST]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_AVX2_SAT].c_str(), accelerationModes[GNA_AVX2_SAT]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_AVX1_FAST].c_str(), accelerationModes[GNA_AVX1_FAST]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_AVX1_SAT].c_str(), accelerationModes[GNA_AVX1_SAT]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_SSE4_2_FAST].c_str(), accelerationModes[GNA_SSE4_2_FAST]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_SSE4_2_SAT].c_str(), accelerationModes[GNA_SSE4_2_SAT]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_GEN_FAST].c_str(), accelerationModes[GNA_GEN_FAST]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_GEN_SAT].c_str(), accelerationModes[GNA_GEN_SAT]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_SW_FAST].c_str(), accelerationModes[GNA_SW_FAST]);
    Log->Message("%s\t%d\n", accelerationNames[GNA_SW_SAT].c_str(), accelerationModes[GNA_SW_SAT]);
}

acceleration AccelerationDetector::GetFastestAcceleration() const
{
    return fastestAcceleration;
}

char const * const GNA::AccelerationDetector::AccelerationToString(acceleration accel)
{
    auto name = accelerationNames.find(accel);
    if (accelerationNames.end() == name)
    {
        accel = NUM_GNA_ACCEL_MODES;
    }
    return accelerationNames[accel].c_str();
}

void AccelerationDetector::UpdateKernelsMap()
{
    if (ACC_SUPPORTED == accelerationModes[GNA_SSE4_2_FAST])
    {
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.affineSingle1Bfull);
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.affineSingle1Bfull);

        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.affineSingle2Bfull);
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.affineSingle2Bfull);

        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.affineSingle1Bal);
        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.affineSingle1Bal);

        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.affineSingle2Bal);
        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.affineSingle2Bal);

        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.diagonal1B);
        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.diagonal1B);

        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.diagonal2B);
        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.diagonal2B);

        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.affineMulti1B);
        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.affineMulti1B);

        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.affineMulti2B);
        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.affineMulti2B);

        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.recurrent1B);
        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_1B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.recurrent1B);

        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.recurrent2B);
        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_2B).emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.recurrent2B);

        AccelerationDetector::ConvolutionKernels.emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.convolution);
        AccelerationDetector::ConvolutionKernels.emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.convolution);

        AccelerationDetector::PoolingKernels.emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.convolutionPooling);
        AccelerationDetector::PoolingKernels.emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.convolutionPooling);

        AccelerationDetector::GmmKernels.at(GNA_MAXMIX8).emplace(GNA_SSE4_2_FAST, gmmKernel_sse4.gmmMaxMix8);
        AccelerationDetector::GmmKernels.at(GNA_MAXMIX8).emplace(GNA_SSE4_2_SAT, gmmKernel_sse4.gmmMaxMix8);

        AccelerationDetector::GmmKernels.at(GNA_MAXMIX16).emplace(GNA_SSE4_2_FAST, gmmKernel_sse4.gmmMaxMix16);
        AccelerationDetector::GmmKernels.at(GNA_MAXMIX16).emplace(GNA_SSE4_2_SAT, gmmKernel_sse4.gmmMaxMix16);

        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX8).emplace(GNA_SSE4_2_FAST, gmmKernel_sse4.gmmMaxMix8ActiveList);
        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX8).emplace(GNA_SSE4_2_SAT, gmmKernel_sse4.gmmMaxMix8ActiveList);

        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX16).emplace(GNA_SSE4_2_FAST, gmmKernel_sse4.gmmMaxMix16ActiveList);
        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX16).emplace(GNA_SSE4_2_SAT, gmmKernel_sse4.gmmMaxMix16ActiveList);

        AccelerationDetector::PwlKernels.emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.pwl);
        AccelerationDetector::PwlKernels.emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.pwl);

        AccelerationDetector::TransposeKernels.emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.transpose);
        AccelerationDetector::TransposeKernels.emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.transpose);

        AccelerationDetector::CopyKernels.emplace(GNA_SSE4_2_FAST, xnnKernel_sse4.copy);
        AccelerationDetector::CopyKernels.emplace(GNA_SSE4_2_SAT, xnnKernel_sse4_sat.copy);
    }

    if (ACC_SUPPORTED == accelerationModes[GNA_AVX1_FAST])
    {
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.affineSingle1Bfull);
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.affineSingle1Bfull);

        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.affineSingle2Bfull);
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.affineSingle2Bfull);

        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.affineSingle1Bal);
        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.affineSingle1Bal);

        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.affineSingle2Bal);
        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.affineSingle2Bal);

        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.diagonal1B);
        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.diagonal1B);

        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.diagonal2B);
        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.diagonal2B);

        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.affineMulti1B);
        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.affineMulti1B);

        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.affineMulti2B);
        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.affineMulti2B);

        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.recurrent1B);
        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.recurrent1B);

        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_FAST, xnnKernel_avx1.recurrent2B);
        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.recurrent2B);

        AccelerationDetector::ConvolutionKernels.emplace(GNA_AVX1_FAST, xnnKernel_avx1.convolution);
        AccelerationDetector::ConvolutionKernels.emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.convolution);

        AccelerationDetector::PoolingKernels.emplace(GNA_AVX1_FAST, xnnKernel_avx1.convolutionPooling);
        AccelerationDetector::PoolingKernels.emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.convolutionPooling);

        AccelerationDetector::GmmKernels.at(GNA_MAXMIX8).emplace(GNA_AVX1_FAST, gmmKernel_avx1.gmmMaxMix8);
        AccelerationDetector::GmmKernels.at(GNA_MAXMIX8).emplace(GNA_AVX1_SAT, gmmKernel_avx1.gmmMaxMix8);

        AccelerationDetector::GmmKernels.at(GNA_MAXMIX16).emplace(GNA_AVX1_FAST, gmmKernel_avx1.gmmMaxMix16);
        AccelerationDetector::GmmKernels.at(GNA_MAXMIX16).emplace(GNA_AVX1_SAT, gmmKernel_avx1.gmmMaxMix16);

        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX8).emplace(GNA_AVX1_FAST, gmmKernel_avx1.gmmMaxMix8ActiveList);
        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX8).emplace(GNA_AVX1_SAT, gmmKernel_avx1.gmmMaxMix8ActiveList);

        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX16).emplace(GNA_AVX1_FAST, gmmKernel_avx1.gmmMaxMix16ActiveList);
        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX16).emplace(GNA_AVX1_SAT, gmmKernel_avx1.gmmMaxMix16ActiveList);

        AccelerationDetector::PwlKernels.emplace(GNA_AVX1_FAST, xnnKernel_avx1.pwl);
        AccelerationDetector::PwlKernels.emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.pwl);

        AccelerationDetector::TransposeKernels.emplace(GNA_AVX1_FAST, xnnKernel_avx1.transpose);
        AccelerationDetector::TransposeKernels.emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.transpose);

        AccelerationDetector::CopyKernels.emplace(GNA_AVX1_FAST, xnnKernel_avx1.copy);
        AccelerationDetector::CopyKernels.emplace(GNA_AVX1_SAT, xnnKernel_avx1_sat.copy);
    }

    if (ACC_SUPPORTED == accelerationModes[GNA_AVX2_FAST])
    {
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.affineSingle1Bfull);
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.affineSingle1Bfull);

        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.affineSingle2Bfull);
        AccelerationDetector::AffineKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.affineSingle2Bfull);

        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.affineSingle1Bal);
        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.affineSingle1Bal);

        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.affineSingle2Bal);
        AccelerationDetector::AffineKernelsAl.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.affineSingle2Bal);

        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.diagonal1B);
        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.diagonal1B);

        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.diagonal2B);
        AccelerationDetector::DiagonalKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.diagonal2B);

        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.affineMulti1B);
        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.affineMulti1B);

        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.affineMulti2B);
        AccelerationDetector::MultibiasKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.affineMulti2B);

        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.recurrent1B);
        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_1B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.recurrent1B);

        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_FAST, xnnKernel_avx2.recurrent2B);
        AccelerationDetector::RecurrentKernels.at(GNA_WEIGHT_2B).emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.recurrent2B);

        AccelerationDetector::ConvolutionKernels.emplace(GNA_AVX2_FAST, xnnKernel_avx2.convolution);
        AccelerationDetector::ConvolutionKernels.emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.convolution);

        AccelerationDetector::PoolingKernels.emplace(GNA_AVX2_FAST, xnnKernel_avx2.convolutionPooling);
        AccelerationDetector::PoolingKernels.emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.convolutionPooling);

        AccelerationDetector::GmmKernels.at(GNA_MAXMIX8).emplace(GNA_AVX2_FAST, gmmKernel_avx2.gmmMaxMix8);
        AccelerationDetector::GmmKernels.at(GNA_MAXMIX8).emplace(GNA_AVX2_SAT, gmmKernel_avx2.gmmMaxMix8);

        AccelerationDetector::GmmKernels.at(GNA_MAXMIX16).emplace(GNA_AVX2_FAST, gmmKernel_avx2.gmmMaxMix16);
        AccelerationDetector::GmmKernels.at(GNA_MAXMIX16).emplace(GNA_AVX2_SAT, gmmKernel_avx2.gmmMaxMix16);

        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX8).emplace(GNA_AVX2_FAST, gmmKernel_avx2.gmmMaxMix8ActiveList);
        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX8).emplace(GNA_AVX2_SAT, gmmKernel_avx2.gmmMaxMix8ActiveList);

        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX16).emplace(GNA_AVX2_FAST, gmmKernel_avx2.gmmMaxMix16ActiveList);
        AccelerationDetector::GmmActiveListKernels.at(GNA_MAXMIX16).emplace(GNA_AVX2_SAT, gmmKernel_avx2.gmmMaxMix16ActiveList);

        AccelerationDetector::PwlKernels.emplace(GNA_AVX2_FAST, xnnKernel_avx2.pwl);
        AccelerationDetector::PwlKernels.emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.pwl);

        AccelerationDetector::TransposeKernels.emplace(GNA_AVX2_FAST, xnnKernel_avx2.transpose);
        AccelerationDetector::TransposeKernels.emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.transpose);

        AccelerationDetector::CopyKernels.emplace(GNA_AVX2_FAST, xnnKernel_avx2.copy);
        AccelerationDetector::CopyKernels.emplace(GNA_AVX2_SAT, xnnKernel_avx2_sat.copy);
    }
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap(WeightMode weightMode, nn_layer_kind layerKind)
{
    throw GnaException{ GNA_CPUTYPENOTSUPPORTED };
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap(WeightMode weightMode)
{
    throw GnaException{ GNA_CPUTYPENOTSUPPORTED };
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap()
{
    throw GnaException{ GNA_CPUTYPENOTSUPPORTED };
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap(gna_gmm_mode)
{
    throw GnaException{ GNA_CPUTYPENOTSUPPORTED };
}

template<>
const std::map<const acceleration, const AffineKernel>&
AccelerationDetector::GetKernelMap<AffineKernel>(WeightMode weightMode, nn_layer_kind layerKind)
{
    switch (layerKind)
    {
    case INTEL_AFFINE:
        /* FALLTHRU */
    case INTEL_RECURRENT:
        return AffineKernels.at(weightMode);
    case INTEL_AFFINE_DIAGONAL:
        return DiagonalKernels.at(weightMode);
    case INTEL_AFFINE_MULTIBIAS:
        return MultibiasKernels.at(weightMode);
    default:
        throw GnaException{ XNN_ERR_LYR_KIND };
    }
}

template<>
const std::map<const acceleration, const AffineActiveListKernel>&
AccelerationDetector::GetKernelMap<AffineActiveListKernel>(WeightMode weightMode)
{
    return AffineKernelsAl.at(weightMode);
}

template<>
const std::map<const acceleration, const RecurrentKernel>&
AccelerationDetector::GetKernelMap<RecurrentKernel>(WeightMode weightMode)
{
    return RecurrentKernels.at(weightMode);
}

template<>
const std::map<const acceleration, const ConvolutionKernel>&
AccelerationDetector::GetKernelMap<ConvolutionKernel>()
{
    return ConvolutionKernels;
}

template<>
const std::map<const acceleration, const ConvolutionPoolingKernel>&
AccelerationDetector::GetKernelMap<ConvolutionPoolingKernel>()
{
    return PoolingKernels;
}

template<>
const std::map<const acceleration, const PwlKernel>&
AccelerationDetector::GetKernelMap<PwlKernel>()
{
    return PwlKernels;
}

template<>
const std::map<const acceleration, const TransposeKernel>&
AccelerationDetector::GetKernelMap<TransposeKernel>()
{
    return TransposeKernels;
}

template<>
const std::map<const acceleration, const CopyKernel>&
AccelerationDetector::GetKernelMap<CopyKernel>()
{
    return CopyKernels;
}

template<>
const std::map<const acceleration, const GmmMaxMix>&
AccelerationDetector::GetKernelMap<GmmMaxMix>(gna_gmm_mode gmmMode)
{
    return GmmKernels.at(gmmMode);
}

template<>
const std::map<const acceleration, const GmmMaxMixActiveList>&
AccelerationDetector::GetKernelMap<GmmMaxMixActiveList>(gna_gmm_mode gmmMode)
{
    return GmmActiveListKernels.at(gmmMode);
}
