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

#define AffineKernelImpl2B KERNEL(AffineKernelImpl2B)
#define AffineActiveListKernelImpl2B KERNEL(AffineActiveListKernelImpl2B)
#define AffineMultiBiasKernelImpl2B KERNEL(AffineMultiBiasKernelImpl2B)
#define RecurrentKernelImpl2B KERNEL(RecurrentKernelImpl2B)
#define DiagonalKernelImpl2B KERNEL(DiagonalKernelImpl2B)

#define AffineActiveListKernelImpl2B1B KERNEL(AffineActiveListKernelImpl2B1B1B)
#define RecurrentKernelImpl2B1B KERNEL(RecurrentKernelImpl2B1B)
#define DiagonalKernelImpl2B1B KERNEL(DiagonalKernelImpl2B1B)
#define AffineKernelImpl2B1B KERNEL(AffineKernelImpl2B1B)
#define AffineMultiBiasKernelImpl2B1B KERNEL(AffineMultiBiasKernelImpl2B1B)
#define TransposeKernelImpl1B KERNEL(TransposeKernelImpl1B)

#define AffineActiveListKernelImpl2B2B KERNEL(AffineActiveListKernelImpl2B2B)
#define RecurrentKernelImpl2B2B KERNEL(RecurrentKernelImpl2B2B)
#define DiagonalKernelImpl2B2B KERNEL(DiagonalKernelImpl2B2B)
#define AffineKernelImpl2B2B KERNEL(AffineKernelImpl2B2B)
#define AffineMultiBiasKernelImpl2B2B KERNEL(AffineMultiBiasKernelImpl2B2B)
#define TransposeKernelImpl2B KERNEL(TransposeKernelImpl2B)

#ifdef __cplusplus
extern "C" {
#endif

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
void AffineKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// uses active outputs list
void AffineActiveListKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// handles multi bias
void AffineMultiBiasKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates recurrent transform on flat input vectors
// (input vectors in N rows, vector elements in K columns)
void RecurrentKernelImpl2B(ExecutionKernelConfig<RecurrentConfig> const * const config);

void DiagonalKernelImpl2B(ExecutionKernelConfig<AffineConfig> const * const config);

void TransposeKernelImpl2B(TransposeConfig const * const transposeConfig);

#if OPT_LEVEL < 2
void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig);
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineActiveListKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void RecurrentKernelImpl2B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void RecurrentKernelImpl2B2B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void DiagonalKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void DiagonalKernelImpl2B2B(ExecutionKernelConfig<AffineConfig> const * const config);
#endif

#if OPT_LEVEL == 7
void AffineKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl2B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
#endif

#ifdef __cplusplus
}
#endif
