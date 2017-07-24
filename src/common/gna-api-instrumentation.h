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

/******************************************************************************
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * Development purposes API Definition
 *
 *****************************************************************************/

#pragma once

#if !defined(DRIVER)

#include "gna-api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _acceleration_all
{
    GNA_HW = GNA_HARDWARE,
    GNA_AUTO_SAT = GNA_AUTO & GNA_HW,
    GNA_AUTO_FAST = GNA_AUTO,
    GNA_SW_SAT = GNA_SOFTWARE & GNA_HW,
    GNA_SW_FAST = GNA_SOFTWARE,
    GNA_GEN_SAT = GNA_GENERIC & GNA_HW,
    GNA_GEN_FAST = GNA_GENERIC,
    GNA_SSE4_2_SAT = GNA_SSE4_2 & GNA_HW,
    GNA_SSE4_2_FAST = GNA_SSE4_2,
    GNA_AVX1_SAT = GNA_AVX1 & GNA_HW,
    GNA_AVX1_FAST = GNA_AVX1,
    GNA_AVX2_SAT = GNA_AVX2 & GNA_HW,
    GNA_AVX2_FAST = GNA_AVX2,
    NUM_GNA_ACCEL_MODES = 0xE,
} gna_acceleration_all;

static_assert(4 == sizeof(gna_acceleration_all), "Invalid size of gna_acceleration_all");

/**
 * GNA HW Scoring Acceleration performance counters.
 * When performance counting is enabled, the total scoring cycles counter is always on.
 * In addition one of several reasons for stall may be measured to allow
 * identifying the bottlenecks in the scoring operation.
 */
typedef enum _gna_hw_perf_encoding
{
    PERF_COUNT_DISABLED = 0,
    COUNT_TOTAL_STALL_CYCLE = 1,
    WAIT_FOR_DMA_COMPLETION = 2,
    WAIT_FOR_MMU_TRANSLATION = 3,
    DESCRIPTOR_FETCH_TIME = 4,
    INPUT_BUFFER_FILL_FROM_MEMORY = 5,
    OUTPUT_BUFFER_FULL_STALL_CYCLES = 6,
    OUTPUT_BUFFER_WAIT_FOR_IOSF_STALL_CYCLES = 7
} gna_hw_perf_encoding;

#endif //DRIVER

#ifndef PERF_TYPE_DEF

#define PERF_TYPE_DEF

/**
 * Time Stamp Counter time type
 */
typedef unsigned long long time_tsc;

static_assert(8 == sizeof(time_tsc), "Invalid size of time_tsc");

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

#if !defined(DRIVER)

/**
 * Adds single request configuration for use with model.
 * Request configurations have to be declared a priori to minimize the time
 * of preparation of request and reduce processing latency.
 * This configuration holds buffers that can be used with consecutive requests
 * to handle asynchronous processing.
 * When request are processed asynchronously each need to have individual
 * Input and output buffers set by this configuration.
 * Configurations can be reused with another request when request
 * with current configuration has been completed and retrieved by GnaRequestWait.
 * I.e. User can create e.g. 8 unique configurations and reuse them
 * with consecutive batches of 8 requests, when batches are enqueued sequentially.
 * NOTE:
 * - Unreleased configurations are released by GNA with corresponding model release.
 *
 * @param modelId           Model, that request configuration will be used with.
 *                          Configuration cannot be shared with other models.
 * @param configId          (out) Request configuration created by GNA.
 * @param hwPerfEncoding    Type of performance statistic.
 * @param perfResults       Buffer to save performance measurements to or NULL to ignore.
 */
GNAAPI intel_gna_status_t GnaRequestConfigEnablePerf(
    gna_request_cfg_id      configId,
    gna_hw_perf_encoding    hwPerfEncoding,
    gna_perf_t*             perfResults);

#ifdef __cplusplus
}
#endif

#endif //DRIVER

