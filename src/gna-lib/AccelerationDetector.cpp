/*
 INTEL CONFIDENTIAL
 Copyright 2018-2020 Intel Corporation.

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
    return static_cast<unsigned long long>(a) | ((static_cast<unsigned long long>(d)) << 32);
}
#define cpuid(info, level) __cpuid_count(level, 0, info[0], info[1], info[2], info[3])
#else
#if defined(__INTEL_COMPILER)
#include <immintrin.h>
#else // __INTEL_COMPILER
#include <intrin.h>
#endif
#define cpuid(info, level) __cpuidex((int*)(info), level, 0)
#endif // __GNUC__

#include "gmm.h"

#include "AccelerationDetector.h"
#include "Logger.h"

#include "gna2-inference-impl.h"

#include <map>
#include <memory>

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

static constexpr auto SAT = true;
static constexpr auto FAST = false;

template<Gna2AccelerationMode accelerationMode, bool hardwareConsistencyEnabled,
    bool isAccelerated>
    KernelMap<VoidKernel>::allocator_type::value_type MakeSingleEntry(KernelType kernel)
{
    return {
        AccelerationMode{accelerationMode, hardwareConsistencyEnabled},
        { GetXnnKernel<isAccelerated ? accelerationMode : Gna2AccelerationModeGeneric, hardwareConsistencyEnabled>(kernel) }
    };
}

template<KernelType generic, KernelType see4x2, KernelType avx1, KernelType avx2,
    bool isSseAccelerated, bool isAvx1Accelerated, bool isAvx2Accelerated>
KernelMap<VoidKernel> MakeAll()
{
    return
    {
        MakeSingleEntry<Gna2AccelerationModeGeneric, SAT, false>(generic),
        MakeSingleEntry<Gna2AccelerationModeGeneric, FAST, false>(generic),
        MakeSingleEntry<Gna2AccelerationModeSse4x2, SAT, isSseAccelerated>(see4x2),
        MakeSingleEntry<Gna2AccelerationModeSse4x2, FAST, isSseAccelerated>(see4x2),
        MakeSingleEntry<Gna2AccelerationModeAvx1, SAT, isAvx1Accelerated>(avx1),
        MakeSingleEntry<Gna2AccelerationModeAvx1, FAST, isAvx1Accelerated>(avx1),
        MakeSingleEntry<Gna2AccelerationModeAvx2, SAT, isAvx2Accelerated>(avx2),
        MakeSingleEntry<Gna2AccelerationModeAvx2, FAST, isAvx2Accelerated>(avx2),
    };
}

template<KernelType genericKernel, KernelType acceleratedKernel>
KernelMap<VoidKernel> MakeAllAccelerated()
{
    return MakeAll<genericKernel, acceleratedKernel, acceleratedKernel, acceleratedKernel,
        true, true, true>();
}

template<KernelType genericKernel, KernelType acceleratedKernel>
KernelMap<VoidKernel> MakeSseAccelerated()
{
    return MakeAll<genericKernel, acceleratedKernel, genericKernel, genericKernel,
        true, false, false>();
}

template<KernelType genericKernel>
KernelMap<VoidKernel> MakeAllGeneric()
{
    return MakeAll<genericKernel, genericKernel, genericKernel, genericKernel,
    false, false, false>();
}

template<Gna2AccelerationMode mode, bool hardwareConsistencyEnabled, typename GmmKernelType>
KernelMap<VoidKernel>::allocator_type::value_type MakeGmm(GmmKernelType kernel)
{
    return {
        AccelerationMode{mode, hardwareConsistencyEnabled},
        { reinterpret_cast<VoidKernel>(kernel) }
    };
}

template<typename GmmKernelType>
KernelMap<VoidKernel> MakeGmm(GmmKernelType kernel1, GmmKernelType kernel2, GmmKernelType kernel3, GmmKernelType kernel4,
    GmmKernelType kernel5, GmmKernelType kernel6, GmmKernelType kernel7, GmmKernelType kernel8)
{
    return
    {
        MakeGmm<Gna2AccelerationModeGeneric, SAT, GmmKernelType>(kernel1),
        MakeGmm<Gna2AccelerationModeGeneric, FAST, GmmKernelType>(kernel2),
        MakeGmm<Gna2AccelerationModeSse4x2, SAT, GmmKernelType>(kernel3),
        MakeGmm<Gna2AccelerationModeSse4x2, FAST, GmmKernelType>(kernel4),
        MakeGmm<Gna2AccelerationModeAvx1, SAT, GmmKernelType>(kernel5),
        MakeGmm<Gna2AccelerationModeAvx1, FAST, GmmKernelType>(kernel6),
        MakeGmm<Gna2AccelerationModeAvx2, SAT, GmmKernelType>(kernel7),
        MakeGmm<Gna2AccelerationModeAvx2, FAST, GmmKernelType>(kernel8),
    };
}

const KernelMap<VoidKernel>& AccelerationDetector::GetKernels(kernel_op operation, KernelMode dataMode)
{
    static const std::map<kernel_op, std::map<KernelMode, KernelMap<VoidKernel>>> kernels =
    {
        { KERNEL_AFFINE, {
            {{Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias },
                MakeAllAccelerated<affineSingle1B2Bfull, affineSingle1Bfull>()},
            {
                { Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllAccelerated<affineSingle2B2Bfull, affineSingle2Bfull>()
            },
            {{ Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<affineSingle1B1Bfull>()
            },
            {{ Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<affineSingle2B1Bfull>()}
        }},
        { KERNEL_AFFINE_AL, {
            {{ Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias },
                MakeAllAccelerated<affineSingle1B2Bal, affineSingle1Bal>()},
            {{ Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllAccelerated<affineSingle2B2Bal, affineSingle2Bal>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<affineSingle1B1Bal>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<affineSingle2B1Bal>()},
        }},
        { KERNEL_AFFINE_MULTIBIAS,{
            {{ Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllAccelerated<affineMulti1B2B, affineMulti1B>()},
            {{ Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllAccelerated<affineMulti2B2B, affineMulti2B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<affineMulti1B1B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<affineMulti2B1B>()},
        }},
        { KERNEL_RECURRENT,{
            {{ Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias },
                MakeSseAccelerated<recurrent1B2B, recurrent1B>()},
            {{ Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeSseAccelerated<recurrent2B2B, recurrent2B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<recurrent1B1B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<recurrent2B1B>()},
        }},
        { KERNEL_AFFINE_DIAGONAL, {
            {{ Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeCompoundBias },
                MakeAllAccelerated<diagonal1B2B, diagonal1B>()},
            {{ Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllAccelerated<diagonal2B2B, diagonal2B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<diagonal1B1B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<diagonal2B1B>()},
        }},
        { KERNEL_TRANSPOSE, {
            {{ Gna2DataTypeInt8},
                MakeAllGeneric<transpose1B>()},
            {{ Gna2DataTypeInt16},
                MakeSseAccelerated<transpose2B, transpose>()},
        }},
        { KERNEL_COPY,{
            {{ Gna2DataTypeInt8 },
                MakeAllGeneric<copy1B>()},
            {{ Gna2DataTypeInt16 },
                MakeAllAccelerated<copy2B, copy>()},
        }},
        { KERNEL_CONVOLUTIONAL, {
            {{ Gna2DataTypeInt16 },
                MakeAllAccelerated<convolution2B, convolution>()},
            {{ Gna2DataTypeInt8 },
                MakeAllGeneric<convolution1B>()},
        }},
        { KERNEL_CONVOLUTIONAL_2D, {
            {{ Gna2DataTypeInt16, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<convolution2D1B2B>()},
            {{ Gna2DataTypeInt16, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<convolution2D2B2B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt8, Gna2DataTypeInt8 },
                MakeAllGeneric<convolution2D1B1B>()},
            {{ Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt8 },
                MakeAllGeneric<convolution2D2B1B>()},
        }},
        { KERNEL_POOLING, {
            {{ Gna2DataTypeInt16, },
                MakeAllAccelerated<convolutionPooling2B, convolutionPooling>()},
            {{ Gna2DataTypeInt8 },
                MakeAllGeneric<convolutionPooling1B>()},
        }},
        { KERNEL_POOLING_2D, {
            {{ Gna2DataTypeInt8 },
                MakeAllGeneric<convolutionPooling2D1B>()},
            {{ Gna2DataTypeInt16 },
                MakeAllGeneric<convolutionPooling2D2B>()},
            {{ Gna2DataTypeInt32 },
                MakeAllGeneric<convolutionPooling2D4B>()},
        }},
        { KERNEL_PWL, {
            {{ Gna2DataTypeInt16 },
                MakeAllAccelerated<pwl, pwl>()},
        }},
        { KERNEL_GMM, {
            {{ Gna2DataTypeUint8, Gna2DataTypeUint8, Gna2DataTypeUint32 },
                MakeGmm(gmmKernel_generic.gmmMaxMix8, gmmKernel_generic.gmmMaxMix8,
                        gmmKernel_sse4.gmmMaxMix8, gmmKernel_sse4.gmmMaxMix8,
                      gmmKernel_avx1.gmmMaxMix8, gmmKernel_avx1.gmmMaxMix8,
                      gmmKernel_avx2.gmmMaxMix8, gmmKernel_avx2.gmmMaxMix8)},
            {{ Gna2DataTypeUint8, Gna2DataTypeUint16, Gna2DataTypeUint32 },
                 MakeGmm(gmmKernel_generic.gmmMaxMix16, gmmKernel_generic.gmmMaxMix16,
                        gmmKernel_sse4.gmmMaxMix16, gmmKernel_sse4.gmmMaxMix16,
                      gmmKernel_avx1.gmmMaxMix16, gmmKernel_avx1.gmmMaxMix16,
                      gmmKernel_avx2.gmmMaxMix16, gmmKernel_avx2.gmmMaxMix16)},
        }},
        { KERNEL_GMM_AL, {
            {{ Gna2DataTypeUint8, Gna2DataTypeUint8, Gna2DataTypeUint32 },
                 MakeGmm(gmmKernel_generic.gmmMaxMix8ActiveList, gmmKernel_generic.gmmMaxMix8ActiveList,
                        gmmKernel_sse4.gmmMaxMix8ActiveList, gmmKernel_sse4.gmmMaxMix8ActiveList,
                      gmmKernel_avx1.gmmMaxMix8ActiveList, gmmKernel_avx1.gmmMaxMix8ActiveList,
                      gmmKernel_avx2.gmmMaxMix8ActiveList, gmmKernel_avx2.gmmMaxMix8ActiveList)},
            {{ Gna2DataTypeUint8, Gna2DataTypeUint16, Gna2DataTypeUint32 },
                MakeGmm(gmmKernel_generic.gmmMaxMix16ActiveList, gmmKernel_generic.gmmMaxMix16ActiveList,
                        gmmKernel_sse4.gmmMaxMix16ActiveList, gmmKernel_sse4.gmmMaxMix16ActiveList,
                      gmmKernel_avx1.gmmMaxMix16ActiveList, gmmKernel_avx1.gmmMaxMix16ActiveList,
                      gmmKernel_avx2.gmmMaxMix16ActiveList, gmmKernel_avx2.gmmMaxMix16ActiveList)},
        }},
    };

    return kernels.at(operation).at(dataMode);
}

AccelerationDetector::AccelerationDetector()
{
    DetectSoftwareAccelerationModes();
}

void AccelerationDetector::DetectSoftwareAccelerationModes()
{
    supportedCpuAccelerations = { Gna2AccelerationModeGeneric };
    // generic, fastest software and auto always supported
    accelerationModes[{Gna2AccelerationModeGeneric, SAT }] = true;
    accelerationModes[{Gna2AccelerationModeGeneric, FAST }] = true;
    accelerationModes[{ Gna2AccelerationModeSoftware, SAT }] = true;
    accelerationModes[{ Gna2AccelerationModeSoftware, FAST }] = true;
    accelerationModes[{ Gna2AccelerationModeAuto, SAT }] = true;
    accelerationModes[{ Gna2AccelerationModeAuto, FAST }] = true;

    unsigned int cpuId[4];           // cpu id string
    unsigned long long xcrFeature;

    cpuid(cpuId, 0);
    const auto largestFunctionId = cpuId[0];

    // get CPU IDs
    cpuid(cpuId, 1);

    // detect SSE4
    // check both SSE4_1, SSE4_2 feature flags (bits 19,20)
    if ((cpuId[2] & SSE4_MASK) == SSE4_MASK)
    {
        accelerationModes[{Gna2AccelerationModeSse4x2, FAST }] = true;
        accelerationModes[{Gna2AccelerationModeSse4x2, SAT }] = true;
        supportedCpuAccelerations.push_back(Gna2AccelerationModeSse4x2);
    }

    if ((cpuId[2] & AVX1_MASK) == AVX1_MASK)
    {
        // check OS has enabled both XMM and YMM state support
        xcrFeature = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        xcrFeature = xcrFeature & XYMM_MASK;
        if (XYMM_MASK == xcrFeature)
        {
            accelerationModes[{Gna2AccelerationModeAvx1, FAST }] = true;
            accelerationModes[{Gna2AccelerationModeAvx1, SAT }] = true;
            supportedCpuAccelerations.push_back(Gna2AccelerationModeAvx1);
        }

        // check AVX2 flag
        if (largestFunctionId >= 7)
        {
            cpuid(cpuId, 7);
            if ((cpuId[1] & AVX2_MASK) == AVX2_MASK)
            {
                accelerationModes[{ Gna2AccelerationModeAvx2, FAST }] = true;
                accelerationModes[{Gna2AccelerationModeAvx2, SAT }] = true;
                supportedCpuAccelerations.push_back(Gna2AccelerationModeAvx2);
            }
        }
    }
}

void AccelerationDetector::PrintAllAccelerationModes() const
{
    for (const auto& modeState : accelerationModes)
    {
        const auto name = modeState.first.GetName();
        Log->Message("%s\t%s\n", name, modeState.second ? "Yes" : "No");
    }
}

const std::vector<Gna2AccelerationMode>& AccelerationDetector::GetSupportedCpuAccelerations() const
{
    return supportedCpuAccelerations;
}
