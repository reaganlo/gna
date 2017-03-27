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

#include "profiler.h"

#ifndef DRIVER
#include <limits.h>
#include <stdlib.h>
#endif // DRIVER

/** 
 * OS time abstraction macros
 */
#if defined(PROFILER_WIN)
// get real time clock value
#define rtcGetTime(t) (int)_ftime64_s(t)
// time_rtc full seconds macro
#define PROFILER_TSEC time
// time_rtc seconds fraction macro
#define PROFILER_TFRAC millitm
// time_rtc seconds fraction resolution
#define PROFILER_TFRAC_RES 1000
#endif // os

#if defined(PROFILE) || defined(PROFILE_DETAILED)

void profilerTscStart(gna_profiler_tsc * const profiler)
{
    int tmp[4];

    if(NULL == profiler) return;

    profiler->passed = 0;
    profiler->stop   = 0;

    __cpuid(tmp, 0);

    profiler->start  = (time_tsc)__rdtsc();
}

void profilerTscStop(gna_profiler_tsc * const profiler)
{
    int tmp[4];

    if(NULL == profiler) return;

    __cpuid(tmp, 0);

    profiler->stop   = (time_tsc)__rdtsc();
    profiler->passed = profiler->stop - profiler->start;
}

void profilerTscStartAccumulate(gna_profiler_tsc * const profiler)
{
    int tmp[4];

    if (NULL == profiler) return;

    profiler->stop = 0;

    __cpuid(tmp, 0);

    profiler->start = (time_tsc)__rdtsc();
}

void profilerTscStopAccumulate(gna_profiler_tsc * const profiler)
{
    int tmp[4];

    if (NULL == profiler) return;

    __cpuid(tmp, 0);

    profiler->stop = (time_tsc)__rdtsc();
    profiler->passed += profiler->stop - profiler->start;
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
    if(NULL == profiler) return;

    profiler->passed.PROFILER_TSEC  = 0;
    profiler->passed.PROFILER_TFRAC = 0;
    profiler->stop.PROFILER_TSEC    = 0;
    profiler->stop.PROFILER_TFRAC   = 0;
    rtcGetTime(&profiler->start);
}

void profilerRtcStop(gna_profiler_rtc * const profiler)
{
    if(NULL == profiler) return;

    rtcGetTime(&profiler->stop);
    profiler->passed = rtcGetTimeDiff(&profiler->start, &profiler->stop);
}

time_tsc profilerRtcGetMilis(gna_profiler_rtc * const profiler)
{
    time_tsc milis = TIME_TSC_MAX;

    if(NULL != profiler)
    {
        // check for milis overflow
        if(TIME_TSC_MAX < (profiler->passed.PROFILER_TFRAC / (PROFILER_TFRAC_RES / 1000)) ) 
            return TIME_TSC_MAX;
        milis = profiler->passed.PROFILER_TFRAC / (PROFILER_TFRAC_RES / 1000);
        // check for milis overflow (simplyfied equation!)
        if((TIME_TSC_MAX / 1000) < (profiler->passed.PROFILER_TSEC))
            return TIME_TSC_MAX;
        milis += 1000 * profiler->passed.PROFILER_TSEC;   
    }
    return milis;
}
#endif // DRIVER

#endif // defined(PROFILE) || defined(PROFILE_DETAILED)
