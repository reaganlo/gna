/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.
    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/******************************************************************************
 *
 * GNA 2.0 API
 *
 * Gaussian Mixture Models and Neural Network Accelerator Module
 * Development purposes API Definition
 *
 *****************************************************************************/

#pragma once

#if !defined(DRIVER)

#if !defined(_WIN32)
#include <assert.h>
#endif

#include "gna-api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * Accelerator (hardware level) scoring request performance results
 */
typedef struct
{
    uint64_t            total;      // # of total cycles spent on scoring in hw
    uint64_t            stall;      // # of stall cycles spent in hw (since scoring)
} gna_perf_hw_t;

static_assert(16 == sizeof(gna_perf_hw_t), "Invalid size of gna_perf_hw_t");

/**
 * Accelerator (driver level) scoring request performance results
 */
typedef struct
{
    uint64_t            startHW;    // time of setting up and issuing HW scoring
    uint64_t            scoreHW;    // time between HW scoring start and scoring complete interrupt
    uint64_t            intProc;    // time of processing scoring complete interrupt
} gna_perf_drv_t;

static_assert(24 == sizeof(gna_perf_drv_t), "Invalid size of gna_perf_drv_t");


/**
 * Accelerator (library level) request absolute timing
 */
typedef struct
{
    uint64_t            start;      // absolute request submit time
    uint64_t            stop;       // absolute processing end time
} gna_perf_total_t;

static_assert(16 == sizeof(gna_perf_total_t), "Invalid size of gna_perf_total_t");

/**
 * Accelerator (library level) scoring request performance results
 */
typedef struct
{
    uint64_t            submit;     // time of score request submit
    uint64_t            preprocess; // time of preprocessing request
    uint64_t            process;    // time of processing score request from submit till done notification
    uint64_t            scoring;    // time of computing scores in software mode
    uint64_t            total;      // time of total scoring - includes time when request is waiting in thread pool
    uint64_t            ioctlSubmit;// time of issuing "start scoring IOCTL"
    uint64_t            ioctlWaitOn;// time of waiting for "start scoring IOCTL" completion
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
 * Enables and configures performance profiler for given request configuration.
 *
 * NOTE:
 * - perfResults need to be within the memory allocated previously by GNAAlloc.
 *
 * @param configId          Request configuration to modify.
 * @param hwPerfEncoding    Type of hardware performance statistic.
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

