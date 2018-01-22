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


#ifndef DRIVER
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#if defined(_WIN32)
#include <Windows.h>
#endif
#else
#if defined(_WIN32)
#include <wdm.h>
#define QueryPerformanceCounter(counter) (*counter = KeQueryPerformanceCounter(NULL))
#endif
#endif // DRIVER

#include "profiler.h"

#if defined(_WIN32)
#define rtcGetTime(t) (int)_ftime64_s(t)
#else 
#define rtcGetTime(t) (int)ftime(t)
#endif

// time_rtc full seconds macro
#define PROFILER_TSEC time
// time_rtc seconds fraction macro
#define PROFILER_TFRAC millitm
// time_rtc seconds fraction resolution
#define PROFILER_TFRAC_RES 1000lu


#if defined(PROFILE) || defined(PROFILE_DETAILED)
#if defined(_WIN32)
void profilerTscStart(gna_profiler_tsc * const profiler)
{
    QueryPerformanceCounter((LARGE_INTEGER*)&profiler->start);
}

void profilerTscStop(gna_profiler_tsc * const profiler)
{
    QueryPerformanceCounter((LARGE_INTEGER*)&profiler->stop);
}

#else
#if defined(__GNUC__) && !defined(__clang__)
static __inline__ unsigned long long __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif

void profilerTscStart(gna_profiler_tsc * const profiler)
{
    profiler->start  = (time_tsc)__rdtsc();
}

void profilerTscStop(gna_profiler_tsc * const profiler)
{
    profiler->stop   = (time_tsc)__rdtsc();
}

#endif

time_tsc profilerGetTscPassed(gna_profiler_tsc const * const profiler)
{
    return profiler->stop - profiler->start;
}

#if !defined(DRIVER)
/**
 * Get time passed between start and stop events
 *
 * @start first time measurement value
 * @stop  end time measurement value
 *
 * @return value of time passed (or zeroed value on error)
 */
time_rtc rtcGetTimeDiff(time_rtc* start, time_rtc* stop)
{
    time_rtc diff = { 0 };

    if(NULL != start && NULL != stop)
    {
        diff.PROFILER_TSEC = stop->PROFILER_TSEC - start->PROFILER_TSEC;
        diff.PROFILER_TFRAC = stop->PROFILER_TFRAC - start->PROFILER_TFRAC;
        // correct time if fraction value is less than second
        if(stop->PROFILER_TFRAC - start->PROFILER_TFRAC < 0)
        {
            diff.PROFILER_TFRAC = PROFILER_TFRAC_RES + stop->PROFILER_TFRAC - start->PROFILER_TFRAC;
            diff.PROFILER_TSEC -= 1;
        }
    }
    return diff;
}

void profilerRtcStart(gna_profiler_rtc * const profiler)
{
    profiler->passed.PROFILER_TSEC  = 0;
    profiler->passed.PROFILER_TFRAC = 0;
    profiler->stop.PROFILER_TSEC    = 0;
    profiler->stop.PROFILER_TFRAC   = 0;
    rtcGetTime(&profiler->start);
}

void profilerRtcStop(gna_profiler_rtc * const profiler)
{
    rtcGetTime(&profiler->stop);
    profiler->passed = rtcGetTimeDiff(&profiler->start, &profiler->stop);
}

time_tsc profilerRtcGetMilis(gna_profiler_rtc * const profiler)
{
    time_tsc milis = TIME_TSC_MAX;

    // check for milis overflow
    if (TIME_TSC_MAX < (profiler->passed.PROFILER_TFRAC / (PROFILER_TFRAC_RES / 1000)))
    {
        return TIME_TSC_MAX;
    }
    milis = profiler->passed.PROFILER_TFRAC / (PROFILER_TFRAC_RES / 1000);

    // check for milis overflow (simplyfied equation!)
    if (TIME_TSC_MAX / 1000 < (uint64_t)profiler->passed.PROFILER_TSEC)
    {
        return TIME_TSC_MAX;
    }
    milis += 1000 * profiler->passed.PROFILER_TSEC;
    return milis;
}
#endif // DRIVER

#endif // defined(PROFILE) || defined(PROFILE_DETAILED)
