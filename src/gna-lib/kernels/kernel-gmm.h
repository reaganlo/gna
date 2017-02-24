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

#pragma once

#include "gmm.h"
#include "KernelMacros.h"
#include "common.h"

#define SHIFT_SPHINX 14
#define LOG_SHIFT_SPHINX 9
#define LOG_THRESH_SPHINX 32768 // was originally 29350

#ifdef INTEL64
#define CVT64_128(a) _mm_cvtsi64_si128(*(int64_t*)(a))
#else
#define CVT64_128(a) _mm_loadl_epi64((__m128i*)(a))
#endif

#ifndef print_m128i
#define print_m128i(reg,regname) {__declspec(align(64)) unsigned int tmp1_[4], i; \
    _mm_store_ps((float*)(tmp1_), (reg));                     \
    printf("%s:", regname);                                             \
    for(i = 0; i < 4; i++) printf("%u ", tmp1_[i]); printf("\n");}
#endif
