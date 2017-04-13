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

#include <stdio.h>
#include <stdlib.h>

#include "gna-api-dev.h"
#include "gna-api-extra.h"

/**
 * Data alignment for intrinsics
 */
#define INTRIN_ALIGN 0x40

/**
 * GNA main memory required alignment size
 */
const uint32_t PAGE_SIZE = 0x1000;

/**
 * Rounds a number up, to the nearest multiple of significance
 * Used for calculating memory sizes of GNA data buffers
 *
 * @param number        Memory size or number to round up.
 * @param significance  The multiple to which number will be rounded.
 * @return Rounded integer value.
 */
inline uint32_t GnaRoundUpMultipleOf(uint32_t number, uint32_t significance)
{
    return (((number + significance - 1) / significance) * significance);
}

/**
 * Rounds a number up, to the nearest multiple of 64
 * Used for calculating memory sizes of GNA data buffers
 *
 * @param number        Memory size or number to round up.
 * @return Rounded integer value.
 */
inline int32_t GnaRoundUpMultipleOf64(uint32_t number)
{
    return GnaRoundUpMultipleOf(number, 64);
}

#if defined(_WIN32)
#   define INLINE               __inline
#   define sleep(s)             { clock_t m=s*1000; clock_t t1=clock(); while(clock()-t1<m); }
#   define _kernel_malloc(a)    _aligned_malloc(a, INTRIN_ALIGN)
#   define _gna_malloc(a)       _aligned_malloc(a, PAGE_SIZE)
#   define _gna_free(a)         _aligned_free(a)
#else
#   define INLINE               inline
#   if defined(USING_GCC)
#       define _gna_malloc(a)   memalign(PAGE_SIZE, a)
#       define _gna_free(a)     free(a)
#   else
#       define _gna_malloc(a)   _mm_malloc(a, PAGE_SIZE)
#       define _gna_free(a)     _mm_free(a)
#   endif // USING_GCC
#endif

#if !defined(UNREFERENCED_PARAMETER)
#define UNREFERENCED_PARAMETER(P) (P)
#endif

// stringify macro
#define STR(X) #X
// stringify macro value
#define VAL(Y) STR(Y)

// TODO: define separate logger class/component

// log message class/component name prefix
#ifndef _COMPONENT_
#define _COMPONENT_
#endif

#if defined(USING_GCC ) && !defined(__FUNCTION__)
#define __FUNCTION__ ""
#endif

// message prefix
#define _PREFIX_    "[IntelGna] " _COMPONENT_ __FUNCTION__ ":" VAL(__LINE__) ": "
#if defined(DUMP_ENABLED) || HW_VERBOSE >= 1 || DEBUG >= 1
// warn message logger decorated with prefix
#define WARN(...)        fprintf(stdout, "WARNING:" _PREFIX_ __VA_ARGS__)
// error message logger decorated with prefix
#define ERR(...)  fprintf(stderr, "ERROR:" _PREFIX_ __VA_ARGS__ )
// error status logger decorated with prefix
#define ERRSTS(status)  ERR("FAILED with status: %d\n", (int)status)
// error message logger decorated with prefix with status name print
#define ERRS(msg, sts, ...)  fprintf(stderr, "ERROR:" _PREFIX_ msg " APISTATUS: %s [%d]\n", __VA_ARGS__, GnaStatusToString(sts), sts )
// log message logger
#define LOG(...)  fprintf(stderr, __VA_ARGS__ )
// log message logger decorated with prefix
#define LOGF(...) fprintf(stderr, _PREFIX_ __VA_ARGS__ )
#else // !LOG_ENABLE
#define WARN(...)
#define ERR(...)
#define ERRSTS(status)
#define ERRS(msg, sts, ...)
#define LOG(...)
#define LOGF(...)
#endif //LOG_ENABLE

/**
 * Structure will hold aligned deinterleaved feature vectors
 * and PWL activation function auxiliary buffers used for performance improvements
 * One structure per thread in thread pool will be created and managed by kernel dispatcher
 */
typedef struct
{
    int16_t *d0;
    int16_t *d1;
    int16_t *d2;
    int16_t *d3;
    int16_t *d4;
    int16_t *d5;
    int16_t *d6;
    int16_t *d7;
    int64_t *pool;
    void    *pwl;
    void    *lookup;
    void    *xBase;
    void    *ySeg;
} KernelBuffers;

/**
 * shorter aliases for official GMM API types
 */
typedef gna_acceleration_all        acceleration;
typedef gna_device_id               device_id;
typedef intel_layer_kind_t          nn_layer_kind;
typedef intel_layer_type_t          nn_layer_type;
typedef intel_compound_bias_t       nn_bias_c;
typedef intel_bias_t                nn_bias_s;
typedef intel_affine_func_t         nn_func_affine;
typedef intel_affine_multibias_func_t nn_func_affine_multi;
typedef intel_pwl_segment_t         nn_pwl_seg;
typedef intel_pwl_func_t            nn_func_pwl;
typedef intel_nnet_layer_t          nn_layer;
typedef intel_affine_layer_t        nn_layer_affine;
typedef intel_affine_multibias_layer_t nn_layer_affine_multi;
typedef intel_recurrent_layer_t     nn_layer_reccurent;
typedef intel_copy_layer_t          nn_layer_copy;
typedef intel_pool_type_t           nn_pool_type;
typedef intel_convolutional_layer_t nn_layer_conv;
typedef gna_perf_t                  perf_t;
typedef gna_perf_drv_t              perf_drv_t;
typedef gna_perf_hw_t               perf_hw_t;

#ifndef STATUS_T_ALIAS
#define STATUS_T_ALIAS
typedef intel_gna_status_t      status_t;
#endif

