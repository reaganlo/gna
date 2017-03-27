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

// check OS flags
#if (defined(_WIN32) || defined(_WIN64) || defined(DRIVER))
#define PROFILER_WIN
#else
#error Invalid preprocessor flags! Cannot determine OS.
#endif // OS flags

#if defined(PROFILER_WIN)
#include <sys/timeb.h>
#include <intrin.h>
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
#define profilerDTscAStart       profilerTscStartAccumulate
#define profilerDTscAStop        profilerTscStopAccumulate
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

#ifndef PERF_TYPE_DEF

#define PERF_TYPE_DEF

/**
 * max value of time_tsc type
 */
#define TIME_TSC_MAX ULLONG_MAX

/**
 * Time Stamp Counter time type
 */
typedef unsigned long long time_tsc;

static_assert(8 == sizeof(time_tsc), "Invalid size of time_tsc");

#if !defined(DRIVER)
/**
 * Real Time Clock time type
 */
#if defined(PROFILER_WIN)
typedef struct __timeb64    time_rtc;
#endif //os
#endif // DRIVER

/**
 * Accelerator (hardware level) scoring request performance results
 */
typedef struct
{
    time_tsc            total;      // # of total cycles spent on scoring in hw
    time_tsc            stall;      // # of stall cycles spent in hw (since scoring)
} gna_perf_hw_t;

static_assert(16 == sizeof(gna_perf_hw_t), "Invalid size of gna_perf_hw_t");

/**
 * Accelerator (driver level) scoring request performance results
 */
typedef struct
{
    time_tsc            startHW;    // time of setting up and issuing HW scoring
    time_tsc            scoreHW;    // time between HW scoring start and scoring complete interrupt
    time_tsc            intProc;    // time of processing scoring complete interrupt
} gna_perf_drv_t;

static_assert(24 == sizeof(gna_perf_drv_t), "Invalid size of gna_perf_drv_t");


/**
 * Accelerator (library level) request absolute timing
 */
typedef struct
{
    time_tsc            start;      // absolute request submit time
    time_tsc            stop;       // absolute processing end time
} gna_perf_total_t;

static_assert(16 == sizeof(gna_perf_total_t), "Invalid size of gna_perf_total_t");

/**
 * Accelerator (library level) scoring request performance results
 */
typedef struct
{
    time_tsc            submit;     // time of score request submit
    time_tsc            preprocess; // time of preprocessing request 
    time_tsc            process;    // time of processing score request from submit till done notification
    time_tsc            scoring;    // time of computing scores in software mode
    time_tsc            total;      // time of total scoring - includes time when request is waiting in thread pool
    time_tsc            ioctlSubmit;// time of issuing "start scoring IOCTL"
    time_tsc            ioctlWaitOn;// time of waiting for "start scoring IOCTL" completion
} gna_perf_lib_t;

static_assert(56 == sizeof(gna_perf_lib_t), "Invalid size of gna_perf_lib_t");

/**
 * Accelerator (overall) scoring request performance results
 */
typedef struct
{
    gna_perf_lib_t lib;       // (library level) performance results
    gna_perf_total_t total;   // (library level) request timing
    gna_perf_drv_t drv;       // (driver level) performance results
    gna_perf_hw_t  hw;        // Accelerator (hardware level) performance results
} gna_perf_t;

static_assert(112 == sizeof(gna_perf_t), "Invalid size of gna_perf_t");

#endif //PERF_TYPE_DEF

/**
 * Timestamp counter profiler
 */
typedef struct
{
    time_tsc            start;      // time value on profiler start
    time_tsc            stop;       // time value on profiler stop
    time_tsc            passed;     // time passed between start and stop
} gna_profiler_tsc;

static_assert(24 == sizeof(gna_profiler_tsc), "Invalid size of gna_profiler_tsc");

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

/**
 * Start TSC profiler
 *
 * @p profiler object to start
 */
void profilerTscStart(gna_profiler_tsc * const profiler);

/**
 * Stop TSC profiler, accumulate passed
 *
 * @p   profiler object to stop
 */
void profilerTscStopAccumulate(gna_profiler_tsc * const profiler);

/**
* Start TSC profiler, does not reset passed time
*
* @p profiler object to start
*/
void profilerTscStartAccumulate(gna_profiler_tsc * const profiler);

/**
* Stop TSC profiler
*
* @p   profiler object to stop
*/
void profilerTscStop(gna_profiler_tsc * const profiler);

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
#endif //DRIVER

#endif //#if defined(PROFILE) || defined(PROFILE_DETAILED)

/******************************************************************************
 * Shorter aliases
 *****************************************************************************/
typedef gna_profiler_tsc      profiler_tsc;
typedef gna_perf_drv_t        perf_drv_t;
typedef gna_perf_hw_t         perf_hw_t;
