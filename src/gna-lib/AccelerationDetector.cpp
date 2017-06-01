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

#ifdef _WIN32
#include <intrin.h>
#endif

#include "AccelerationDetector.h"
#include "GnaException.h"
#include "Logger.h"

using std::array;
using std::make_shared;
using std::map;
using std::shared_ptr;

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
    // Basic, CNN,   GMM,  GMMLayer, MultiBias, L1Dist, L2Dist, ComputerVision, Layer8K
{ GNA_CANNONLAKE,    {true, false, true, false,    false,     false,  false,  false, false} },
{ GNA_GEMINILAKE,    {true, true,  true, false,    false,     false,  false,  false, false} },
{ GNA_ICELAKE,       {true, true,  true, false,    false,     false,  false,  false, false} },
{ GNA_TIGERLAKE,     {true, true,  true, true,     true,      false,  false,  false, true } },
{ GNA_LAKEFIELD,     {true, true,  true, true,     true,      false,  false,  false, true } },
{ GNA_SUE_CREEK,     {true, true,  true, false,    false,     false,  false,  false, false} },
{ GNA_SUE_CREEK_2,   {true, true,  true, true,     true,      false,  false,  false, true } }
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
        Log->Message("Hardware not detected.\n");
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
    switch (layerType)
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

bool AccelerationDetector::HasFeature(GnaFeature feature) const
{
    const auto& deviceFeatureMap = gnaFeatureMap.at(deviceCapabilities.device_type);
    return deviceFeatureMap[feature];
}

AccelerationDetector::AccelerationDetector() :
    fastestAcceleration{ GNA_GEN_FAST }
{
    // generic, fastest software and auto always supported
    accelerationModes[GNA_GEN_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_GEN_FAST] = ACC_SUPPORTED;
    accelerationModes[GNA_SW_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_SW_FAST] = ACC_SUPPORTED;
    accelerationModes[GNA_AUTO_SAT] = ACC_SUPPORTED;
    accelerationModes[GNA_AUTO_FAST] = ACC_SUPPORTED;

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
        accelerationModes[GNA_SSE4_2_SAT] = ACC_SUPPORTED;
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
            accelerationModes[GNA_AVX1_SAT] = ACC_SUPPORTED;
            fastestAcceleration = GNA_AVX1_FAST;

            // check AVX2 flag, bit 5
            if (AVX2_MASK & avx2)
            {
                accelerationModes[GNA_AVX2_FAST] = ACC_SUPPORTED;
                accelerationModes[GNA_AVX2_SAT] = ACC_SUPPORTED;
                fastestAcceleration = GNA_AVX2_FAST;
            }
        }
    }

    Log->Message("GNA_HW        %d\n", accelerationModes[GNA_HW]);
    Log->Message("GNA_AUTO_FAST %d\n", accelerationModes[GNA_AUTO_FAST]);
    Log->Message("GNA_AUTO_SAT  %d\n", accelerationModes[GNA_AUTO_SAT]);
    Log->Message("GNA_AVX2_FAST %d\n", accelerationModes[GNA_AVX2_FAST]);
    Log->Message("GNA_AVX2_SAT  %d\n", accelerationModes[GNA_AVX2_SAT]);
    Log->Message("GNA_AVX1_FAST %d\n", accelerationModes[GNA_AVX1_FAST]);
    Log->Message("GNA_AVX1_SAT  %d\n", accelerationModes[GNA_AVX1_SAT]);
    Log->Message("GNA_SSE4_FAST %d\n", accelerationModes[GNA_SSE4_2_FAST]);
    Log->Message("GNA_SSE4_SAT  %d\n", accelerationModes[GNA_SSE4_2_SAT]);
    Log->Message("GNA_GEN_FAST  %d\n", accelerationModes[GNA_GEN_FAST]);
    Log->Message("GNA_GEN_SAT   %d\n", accelerationModes[GNA_GEN_SAT]);
    Log->Message("GNA_GEN_FAST  %d\n", accelerationModes[GNA_SW_FAST]);
    Log->Message("GNA_GEN_SAT   %d\n", accelerationModes[GNA_SW_SAT]);
}

acceleration AccelerationDetector::GetFastestAcceleration() const
{
    return fastestAcceleration;
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
    throw GnaException{ GNA_UNKNOWN_ERROR };
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap(WeightMode weightMode)
{
    throw GnaException{ GNA_UNKNOWN_ERROR };
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap()
{
    throw GnaException{ GNA_UNKNOWN_ERROR };
}

template<typename T>
const std::map<const acceleration, const T>& AccelerationDetector::GetKernelMap(gna_gmm_mode)
{
    throw GnaException{ GNA_UNKNOWN_ERROR };
}

template<>
const std::map<const acceleration, const AffineKernel>&
AccelerationDetector::GetKernelMap<AffineKernel>(WeightMode weightMode, nn_layer_kind layerKind)
{
    switch (layerKind)
    {
    case INTEL_AFFINE:
        return AffineKernels.at(weightMode);
    case INTEL_AFFINE_DIAGONAL:
        return DiagonalKernels.at(weightMode);
    case INTEL_AFFINE_MULTIBIAS:
        return MultibiasKernels.at(weightMode);
    default:
        throw GnaException{ GNA_UNKNOWN_ERROR };
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
