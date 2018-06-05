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

#include <sys/timeb.h>
#include "gna-api-instrumentation.h"

#if defined(_WIN32)
#if !defined(_MSC_VER)
#include <immintrin.h>
#else
#include <intrin.h>
#endif
#else
#include <mmintrin.h>
#endif // os

// enables or disables simple profiling
#if defined(PROFILE) || defined(PROFILE_DETAILED)
#undef PROFILE
#define PROFILE         1
#define PROFILE_(...)   __VA_ARGS__
#else
#define PROFILE_(...)
// disable basic profiling procedures
#define profilerTscStart(...)
#define profilerTscStop(...)
#define profilerTscGetMicros(...) TIME_TSC_MAX
#define profilerRtcStart(...)
#define profilerRtcStop(...)
#define profilerRtcGetMilis(...)  TIME_TSC_MAX
#endif // defined(PROFILE) || defined(PROFILE_DETAILED)

// enables or disables detailed profiling
#if defined(PROFILE) && defined(PROFILE_DETAILED)
#define PROFILE_D_(...) __VA_ARGS__
// enable detailed profiling procedures
#define profilerDTscStart        profilerTscStart
#define profilerDTscStop         profilerTscStop
#define profilerDTscGetMicros    profilerTscGetMicros
#if !defined(DRIVER)
#define profilerDRtcStart        profilerRtcStart
#define profilerDRtcStop         profilerRtcStop
#define profilerDRtcGetMilis     profilerRtcGetMilis
#endif // DRIVER
#else
#define PROFILE_D_(...)
// disable detailed profiling procedures
#define profilerDTscStart(...)
#define profilerDTscStop(...)
#define profilerDTscGetMicros(...) TIME_TSC_MAX
#if !defined(DRIVER)
#define profilerDRtcStart(...)
#define profilerDRtcStop(...)
#define profilerDRtcGetMilis(...)  TIME_TSC_MAX
#endif // DRIVER
#endif // defined(PROFILE) && defined(PROFILE_DETAILED)

// enables or disables profile print macro
#if defined(PROFILE_PRINT) 
#define PROFILE_PRINT_      PROFILE_
#define PROFILE_PRINT_D_    PROFILE_D_
#else
#define PROFILE_PRINT_(...)
#define PROFILE_PRINT_D_(...)
#endif // defined(PROFILE_PRINT) 

/**
 * max value of time_tsc type
 */
#define TIME_TSC_MAX ULLONG_MAX

#if !defined(DRIVER)
/**
 * Real Time Clock time type
 */
#if defined(_WIN32)
typedef struct __timeb64    time_rtc;
#else
typedef struct timeb        time_rtc;
#endif
#endif // DRIVER

/**
 * Timestamp counter profiler
 */
typedef struct
{
    time_tsc            start;      // time value on profiler start
    time_tsc            stop;       // time value on profiler stop
} gna_profiler_tsc;

static_assert(16 == sizeof(gna_profiler_tsc), "Invalid size of gna_profiler_tsc");

#if !defined(DRIVER)
/**
 * Realtime clock profiler
 */
typedef struct
{
    time_rtc            start;      // time value on profiler start
    time_rtc            stop;       // time value on profiler stop
    time_rtc            passed;     // time passed between start and stop
} gna_profiler_rtc;
#endif //DRIVER

#if defined(PROFILE) || defined(PROFILE_DETAILED)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Start TSC profiler
 *
 * @p profiler object to start
 */
void profilerTscStart(gna_profiler_tsc * const profiler);

/**
* Stop TSC profiler
*
* @p   profiler object to stop
*/
void profilerTscStop(gna_profiler_tsc * const profiler);

/**
 * Get TSC profiler ticks passed
 */
time_tsc profilerGetTscPassed(gna_profiler_tsc const * const profiler);

#if !defined(DRIVER)
/**
 * Start RTC profiler
 *
 * NOTE: available resolution is 10-15ms
 *
 * @p profiler object to start
 */
void profilerRtcStart(gna_profiler_rtc * const profiler);

/**
 * Stop RTC profiler
 *
 * NOTE: available resolution is 10-15ms
 *
 * @p   profiler object to stop
 */
void profilerRtcStop(gna_profiler_rtc * const profiler);

/**
 * Get passed miliseconds
 *
 * NOTE: available resolution is 10-15ms
 *
 * @p       stopped profiler object
 * @return  passed time in miliseconds (or TIME_TSC_MAX if p is invalid)
 */
time_tsc profilerRtcGetMilis(gna_profiler_rtc * const profiler);
#ifdef __cplusplus
}
#endif
#endif //DRIVER

#endif //#if defined(PROFILE) || defined(PROFILE_DETAILED)

/******************************************************************************
 * Shorter aliases
 *****************************************************************************/
typedef gna_profiler_tsc      profiler_tsc;
typedef gna_perf_drv_t        perf_drv_t;
typedef gna_perf_hw_t         perf_hw_t;
