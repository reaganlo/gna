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

#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#endif

#include "gna-api-types-xnn.h"
#include "gna-api-types-gmm.h"
#include "gna-api-instrumentation.h"

/**
 * Data alignment for intrinsics
 */
#define INTRIN_ALIGN 0x40

/**
 * GNA main memory required alignment size
 */
const uint32_t PAGE_SIZE = 0x1000;

#define _gna_malloc(a)    _mm_malloc(a, PAGE_SIZE)
#define _kernel_malloc(a) _mm_malloc(a, INTRIN_ALIGN)
#define _gna_free(a)      _mm_free(a)

#if !defined(UNREFERENCED_PARAMETER)
#define UNREFERENCED_PARAMETER(P) ((void)(P))
#endif

// Enable safe functions compatibility
#if defined(__STDC_SECURE_LIB__)
#define __STDC_WANT_SECURE_LIB__ 1
#elif defined(__STDC_LIB_EXT1__)
#define STDC_WANT_LIB_EXT1 1
#else
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
		memcpy(_Destination, _Source, _SourceSize);\
		UNREFERENCED_PARAMETER(_DestinationSize);\
	} while(0);
#endif

/**
 * shorter aliases for official GMM API types
 */
typedef gna_acceleration_all        acceleration;
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

#ifndef STATUS_T_ALIAS
#define STATUS_T_ALIAS
typedef intel_gna_status_t      status_t;
#endif
