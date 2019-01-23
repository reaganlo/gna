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

#include <algorithm>
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

std::map<gna_device_version, const GnaHardwareCapabiities> gnaCapsMap = {
    { GNA_KBL,
        {GMM_DEVICE,
        {false, false, true, false, false, false,  false,  false, false, false, false },
        6,
        {{1, 8}},
        4,
        0,
        0,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_SKL,
        {GMM_DEVICE,
        {false, false, true, false, false, false,  false,  false, false, false, false },
        6,
        {{1, 8}},
        4,
        0,
        0,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_CNL,
        {GNA_0_9,
        {true, false, true, false,    false,     false,  false,  false, false, false, false },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_GLK,
        {GNA_1_0,
        {true, true,  true, false,    false,     false,  false,  false, false, false, false },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_ICL,
        {GNA_1_0,
        {true, true,  true, false,    false,     false,  false,  false, false, false, false },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_SUE_CREEK,
        {GNA_1_0_EMBEDDED,
        {true, true,  true, false,    false,     false,  false,  false, false, false, false },
        3,
        {{2, 8}},
        4,
        1,
        1,
        {6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144},},
    },
    { GNA_TGL,
        {GNA_2_0,
        {true, true,  true, true,     true,      false,  false,  false, true,  true, false  },
        6,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_JELLYFISH,
        {GNA_2_1_EMBEDDED,
        {true, true,  true, true,     true,      false,  false,  false, true,  true, false },
        3,
        {{2, 8}},
        4,
        1,
        1,
        {12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},},
    },
    { GNA_ADL,
        {GNA_3_0,
        {true, true,  true, true,     true,      false,  false,  false, true, true, true },
        8,
        {{1, 16}, {2, 8}},
        4,
        2,
        16,
        {},},
    },
    { GNA_ACE_EMBEDDED,
        {GNA_3_0_EMBEDDED,
        {true, true,  true, true,     true,      false,  false,  false, true,  true, true },
        8,
        {{1, 16}, {2, 8}},
        4,
        2,
        16,
        {},},
    },
    { GNA_ACE_ANNA,
        {GNA_3_1_AUTONOMUS,
        {true, true,  true, true,     true,      false,  false,  false, true,  true, true },
        2,
        {{1, 16}, {2, 8}},
        4,
        2,
        16,
        {},},
    },
};

std::map<kernel_op, std::map<KernelMode, KernelMap<VoidKernel>>>
AccelerationDetector::Kernels = {
    { KERNEL_AFFINE,
    {
        {
            {GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B2Bfull } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B2Bfull } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.affineSingle1Bfull } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.affineSingle1Bfull } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B2Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B2Bfull } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B2Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B2Bfull } }

            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B2Bfull } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B2Bfull } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.affineSingle2Bfull } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.affineSingle2Bfull } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B2Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B2Bfull } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B2Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B2Bfull } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bfull } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bfull } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bfull } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bfull } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bfull } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bfull } }

            }
        },
        {
            { GNA_INT8 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bfull } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bfull } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bfull } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bfull } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bfull } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bfull } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bfull } }
            }
        }
    }
    },
    { KERNEL_AFFINE_AL,
    {
        {
            { GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B2Bal } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B2Bal } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.affineSingle1Bal } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.affineSingle1Bal } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B2Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B2Bal } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B2Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B2Bal } }
            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B2Bal } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B2Bal } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.affineSingle2Bal } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.affineSingle2Bal } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B2Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B2Bal } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B2Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B2Bal } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bal } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bal } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bal } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bal } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bal } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle1B1Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle1B1Bal } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bal } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bal } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bal } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bal } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bal } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineSingle2B1Bal } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineSingle2B1Bal } }
            }
        }
    }
    },
    { KERNEL_AFFINE_MULTIBIAS,
    {
        {
            { GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti1B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti1B2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.affineMulti1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.affineMulti1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti1B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti1B2B } }
            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.affineMulti2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.affineMulti2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B2B } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti1B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti1B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti1B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti1B1B } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.affineMulti2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.affineMulti2B1B } }
            }
        }
    }
    },
    { KERNEL_RECURRENT,
    {
        {
            { GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.recurrent1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.recurrent1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B2B } }
            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.recurrent2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.recurrent2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B2B } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent1B1B } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.recurrent2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.recurrent2B1B } }
            }
        },
    }
    },
    { KERNEL_AFFINE_DIAGONAL,
    {
        {
            { GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_RICH_FORMAT },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.diagonal1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.diagonal1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B2B } }
            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.diagonal2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.diagonal2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B2B } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal1B1B } }
            }
        },
        {
            { GNA_INT8 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.diagonal2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.diagonal2B1B } }
            }
        },
    }
    },
    { KERNEL_TRANSPOSE,
    {
        {
            { GNA_INT8},
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.transpose1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.transpose1B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.transpose1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.transpose1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.transpose1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.transpose1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.transpose1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.transpose1B } }
            }
        },
        {
            { GNA_INT16},
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.transpose2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.transpose2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.transpose2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.transpose2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.transpose2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.transpose2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.transpose2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.transpose2B } }
            }
        }
    }
    },
    { KERNEL_COPY,
    {
        {
            { GNA_INT8 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.copy1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.copy1B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.copy1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.copy1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.copy1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.copy1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.copy1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.copy1B } }
            }
        },
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.copy2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.copy2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.copy2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.copy2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.copy2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.copy2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.copy2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.copy2B } }
            }
        }
    }
    },
    { KERNEL_CONVOLUTIONAL,
    {
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolution2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.convolution } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.convolution } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2B } }
            }
        },
        {
            { GNA_INT8 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolution1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolution1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution1B } }
            }
        }
    }
    },
    { KERNEL_CONVOLUTIONAL_2D,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B2B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B2B } }
            }
        },
        {
            { GNA_INT16, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B2B } },
               { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B2B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B2B } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D1B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D1B1B } }
            }
        },
        {
            { GNA_INT8, GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolution2D2B1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolution2D2B1B } }
            }
        }
    }
    },
        // TODO:3:CNN2D add pooling kernels
    { KERNEL_POOLING,
    {
        {
            { GNA_INT16, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2B } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.convolutionPooling } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.convolutionPooling } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2B } }
            }
        },
        {
            { GNA_INT8, GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling1B } }
            }
        }
    }
    },
    { KERNEL_POOLING_2D,
    {
        {
            { GNA_INT8 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D1B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D1B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D1B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D1B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D1B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D1B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D1B } }
            }
        },
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D2B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D2B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D2B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D2B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D2B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D2B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D2B } }
            }
        },
        {
            { GNA_INT32 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } }
            }
        },
         {
            { GNA_DATA_ACTIVATION_DISABLED },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } },

                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.convolutionPooling2D4B } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.convolutionPooling2D4B } }
            }
        },
    }
    },
    { KERNEL_PWL,
    {
        {
            { GNA_INT16 },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)xnnKernel_generic_sat.pwl } },
                { { GNA_GEN_FAST },{ (VoidKernel)xnnKernel_generic.pwl } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)xnnKernel_sse4_sat.pwl } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)xnnKernel_sse4.pwl } },

                { { GNA_AVX1_SAT },{ (VoidKernel)xnnKernel_generic_sat.pwl } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.pwl } },
                { { GNA_AVX2_SAT },{ (VoidKernel)xnnKernel_generic_sat.pwl } },
                { { GNA_AVX1_FAST },{ (VoidKernel)xnnKernel_generic.pwl } }
            }
        }
    }
    },
    { KERNEL_GMM,
    {
        {
            { GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8 } },
                { { GNA_GEN_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8 } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix8 } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix8 } },

                { { GNA_AVX1_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8 } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8 } },
                { { GNA_AVX2_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8 } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8 } }
            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16 } },
                { { GNA_GEN_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16 } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix16 } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix16 } },

                { { GNA_AVX1_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16 } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16 } },
                { { GNA_AVX2_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16 } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16 } }
            }
        }
    }
    },
    { KERNEL_GMM_AL,
    {
        {
            { GNA_INT16 ,GNA_INT8, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8ActiveList } },
                { { GNA_GEN_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8ActiveList } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix8ActiveList } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix8ActiveList } },

                { { GNA_AVX1_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8ActiveList } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8ActiveList } },
                { { GNA_AVX2_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8ActiveList } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix8ActiveList } }
            }
        },
        {
            { GNA_INT16 ,GNA_INT16, GNA_BIAS_MODE_1_2_4B },
            {
                { { GNA_GEN_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16ActiveList } },
                { { GNA_GEN_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16ActiveList } },
                { { GNA_SSE4_2_SAT },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix16ActiveList } },
                { { GNA_SSE4_2_FAST },{ (VoidKernel)gmmKernel_sse4.gmmMaxMix16ActiveList } },

                { { GNA_AVX1_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16ActiveList } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16ActiveList } },
                { { GNA_AVX2_SAT },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16ActiveList } },
                { { GNA_AVX1_FAST },{ (VoidKernel)gmmKernel_generic.gmmMaxMix16ActiveList } }
            }
        }
    }
    }
};

uint32_t AccelerationDetector::bufferElementsForSw[2][XNN_N_GROUP_MAX] = {};

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

gna_device_version AccelerationDetector::GetDeviceVersion(gna_device_generation generation)
{
    auto type = std::find_if(gnaCapsMap.cbegin(), gnaCapsMap.cend(),
        [generation](const std::pair<const gna_device_version, const GnaHardwareCapabiities>& deviceVer)
            {
                return deviceVer.second.Generation == generation;
            });
    return type->first;
}

uint32_t AccelerationDetector::GetComputeEngineCount(gna_device_version hwId)
{
    return gnaCapsMap.at(hwId).ComputeEngineCount;
}

uint32_t AccelerationDetector::GetBufferSizeInKB(gna_device_version hwId)
{
    auto caps = gnaCapsMap.at(hwId);
    return caps.ComputeEngineCount * caps.BufferSizesPerCEInKB;
}

uint32_t AccelerationDetector::GetBufferElementCount(gna_device_version hwId, uint32_t grouping,
    uint32_t inputPrecision)
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

void AccelerationDetector::discoverHardware()
{
    accelerationModes[GNA_HW] = ACC_NOTSUPPORTED;
    try
    {
        ioctlSender.Open();
        deviceCapabilities = ioctlSender.GetDeviceCapabilities();

        if (deviceCapabilities.hwId == GNA_ADL)
            deviceCapabilities.hwInBuffSize = 32; //TODO:3: remove when ADL bug with input buffer will be fixed


        Expect::Equal((size_t)1, gnaCapsMap.count(deviceCapabilities.hwId), GNA_DEVNOTFOUND);
        Expect::Equal(deviceCapabilities.hwInBuffSize, GetBufferSizeInKB(deviceCapabilities.hwId),
            status_t::GNA_ERR_INVALID_DEVICE_VERSION);
        accelerationModes[GNA_HW] = ACC_SUPPORTED;

    }
    catch (GnaException&)
    {
        accelerationModes[GNA_HW] = ACC_NOTSUPPORTED;
        Log->Message("No compatible hardware detected.\n");
    }
    setHwCompatibilityMode(deviceCapabilities.hwId);
}

void AccelerationDetector::setHwCompatibilityMode(gna_device_version hwId)
{
    for (uint32_t p = 0; p < 2; p++)
    {
        for (uint32_t i = 0; i < 8; i++)
        {
            bufferElementsForSw[p][i] = GetBufferElementCount(hwId, i + 1, p + 1);
        }
    }

    setHwCompatibilityMode_generic(bufferElementsForSw);
    setHwCompatibilityMode_sse4(bufferElementsForSw);
    setHwCompatibilityMode_avx1(bufferElementsForSw);
    setHwCompatibilityMode_avx2(bufferElementsForSw);
    setHwCompatibilityMode_generic_sat(bufferElementsForSw);
    setHwCompatibilityMode_sse4_sat(bufferElementsForSw);
    setHwCompatibilityMode_avx1_sat(bufferElementsForSw);
    setHwCompatibilityMode_avx2_sat(bufferElementsForSw);
}


gna_device_version AccelerationDetector::GetDeviceVersion() const
{
    return deviceCapabilities.hwId;
}

bool AccelerationDetector::IsHardwarePresent() const
{
    return ACC_SUPPORTED == accelerationModes.at(GNA_HW);
}

bool AccelerationDetector::IsLayerSupported(nn_operation operation) const
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

bool AccelerationDetector::HasFeature(GnaFeature feature) const
{
    if (!IsHardwarePresent())
        return false;

    const auto& caps = gnaCapsMap.at(deviceCapabilities.hwId);
    return caps.Features.at(feature);
}

AccelerationDetector::AccelerationDetector(IoctlSender &senderIn) :
    deviceCapabilities{GetBufferSizeInKB(GNA_ADL), 0, GNA_ADL},
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

char const * AccelerationDetector::AccelerationToString(acceleration accel)
{
    auto name = accelerationNames.find(accel);
    if (accelerationNames.end() == name)
    {
        accel = NUM_GNA_ACCEL_MODES;
    }
    return accelerationNames[accel].c_str();
}

