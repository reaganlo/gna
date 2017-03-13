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

#include "common.h"

namespace GNA
{

typedef
status_t
(*AffineTransformFn)(
    const nn_layer*          pLayer,
    const uint32_t*          pActiveIndices,
          uint32_t           nActiveIndices,
          uint32_t*          nSaturated,
          KernelBuffers*     fvBuffers);

typedef
status_t
(*AffMBiasTransformFn)(
    const nn_layer*          pLayer,
    const uint32_t*          pActiveIndices,
          uint32_t           nActiveIndices,
          uint32_t*          nSaturated,
          KernelBuffers*     fvBuffers);

typedef
status_t
(*DiagonalTransformFn)(
    const nn_layer* pLayer,
          uint32_t* nSaturated);

typedef
void
(*PWLTransformFn)(
    const nn_layer*     pLayer,
          uint32_t      nRowBegin,
          uint32_t      nRowEnd,
          uint32_t      nColBegin,
          uint32_t      nColEnd,
          uint32_t*     nSaturated,
          void*         pwlBuff);

typedef
status_t
(*RecurrentTransformFn)(
    const nn_layer*     pLayer,
          uint32_t*     nSaturated,
          void*         pwlBuff);

typedef
status_t
(*TransposeFn)(
    const nn_layer*     pLayer);

typedef
status_t
(*CopyFn)(
    const nn_layer*     pLayer);

typedef
status_t
(*ConvTransformFn)(
    const nn_layer*     pLayer,
          uint32_t*     nSaturated,
          void*         pwlBuff,
          int64_t*      pool);

/**
 * Xnn kernel provider
 *
 *  Contains XNN kernel function pointers for selected acceleration
 */
typedef struct _XnnKernel
{
    AffineTransformFn   affine;     // apply affine transform function
    AffMBiasTransformFn affineMbias;// apply affine transform function
    DiagonalTransformFn diagonal;   // apply affine transform function
    PWLTransformFn      pwl;        // apply piecewise linear transform function
    RecurrentTransformFn recurrent; // apply recurrent transform function
    TransposeFn         transpose;  // apply transpose transform function
    CopyFn              copy;       // apply copy function
    ConvTransformFn     conv;       // apply convolutional transform function

} XnnKernel;                        // Xnn kernel provider

/**
 * Export list of available Xnn kernels providers
 */

/** FAST VERSIONS */
/** generic Xnn kernel provider */
extern XnnKernel xnnKernel_generic;

/** sse4.2 accelerated Xnn kernel provider */
extern XnnKernel xnnKernel_sse4;

/** avx1 accelerated Xnn kernel provider */
extern XnnKernel xnnKernel_avx1;

/** avx2 accelerated Xnn kernel provider */
extern XnnKernel xnnKernel_avx2;

/** SATURATED VERSIONS */
/** generic Xnn kernel provider */
extern XnnKernel xnnKernel_generic_sat;

/** sse4.2 accelerated Xnn kernel provider */
extern XnnKernel xnnKernel_sse4_sat;

/** avx1 accelerated Xnn kernel provider */
extern XnnKernel xnnKernel_avx1_sat;

/** avx2 accelerated Xnn kernel provider */
extern XnnKernel xnnKernel_avx2_sat;

}
