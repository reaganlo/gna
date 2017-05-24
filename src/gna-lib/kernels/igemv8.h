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

#include "KernelArguments.h"
#include "KernelMacros.h"

#define AffineKernelImpl1B KERNEL(AffineKernelImpl1B)
#define AffineActiveListKernelImpl1B KERNEL(AffineActiveListKernelImpl1B)
#define AffineMultiBiasKernelImpl1B KERNEL(AffineMultiBiasKernelImpl1B)
#define AffineMultiBiasActiveListKernelImpl1B KERNEL(AffineMultiBiasActiveListKernelImpl1B)
#define RecurrentKernelImpl1B KERNEL(RecurrentKernelImpl1B)
#define DiagonalKernelImpl1B KERNEL(DiagonalKernelImpl1B)

#ifdef __cplusplus
extern "C" {
#endif

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
void AffineKernelImpl1B(AffineConfig const * const config);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// uses active outputs list
void AffineActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// handles multi bias
void AffineMultiBiasKernelImpl1B(AffineConfig const * const config);

// Calculates affine transform on interleaved input vectors
//  (input vectors in N columns, vector elements in K rows)
//  uses active outputs list
// handles multi bias
void AffineMultiBiasActiveListKernelImpl1B(AffineConfig const * const config, AffineConfigAl const * const al);

// Calculates recurrent transform on flat input vectors
// (input vectors in N rows, vector elements in K columns)
void RecurrentKernelImpl1B(RecurrentConfig const * const config);

void DiagonalKernelImpl1B(AffineConfig const * const config);

#ifdef __cplusplus
}
#endif
