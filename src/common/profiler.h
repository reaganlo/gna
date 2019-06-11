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
#include <chrono>
#include <time.h>

#include "gna2-instrumentation-api.h"

#if defined(_WIN32)
#if !defined(_MSC_VER)
#include <immintrin.h>
#else
#include <intrin.h>
#endif
#else
#include <mmintrin.h>
#endif // os

using chronoClock = std::chrono::high_resolution_clock;
using chronoUs = std::chrono::microseconds;
using chronoMs = std::chrono::milliseconds;

void getTsc(uint64_t * const result);

typedef struct
{
    uint64_t            total;      // # of total cycles spent on scoring in hw
    uint64_t            stall;      // # of stall cycles spent in hw (since scoring)
} gna_perf_hw_t;

typedef struct
{
    uint64_t            startHW;    // time of setting up and issuing HW scoring
    uint64_t            scoreHW;    // time between HW scoring start and scoring complete interrupt
    uint64_t            intProc;    // time of processing scoring complete interrupt
} gna_perf_drv_t;

typedef gna_perf_hw_t perf_hw_t;
typedef gna_perf_drv_t perf_drv_t;