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
#define RecurrentKernelImpl1B KERNEL(RecurrentKernelImpl1B)
#define DiagonalKernelImpl1B KERNEL(DiagonalKernelImpl1B)

#define AffineKernelImpl1B1B KERNEL(AffineKernelImpl1B1B)
#define AffineActiveListKernelImpl1B1B KERNEL(AffineActiveListKernelImpl1B1B)
#define AffineMultiBiasKernelImpl1B1B KERNEL(AffineMultiBiasKernelImpl1B1B)
#define RecurrentKernelImpl1B1B KERNEL(RecurrentKernelImpl1B1B)
#define DiagonalKernelImpl1B1B KERNEL(DiagonalKernelImpl1B1B)
#define TransposeKernelImpl1B KERNEL(TransposeKernelImpl1B)

#define AffineKernelImpl1B2B KERNEL(AffineKernelImpl1B2B)
#define AffineActiveListKernelImpl1B2B KERNEL(AffineActiveListKernelImpl1B2B)
#define AffineMultiBiasKernelImpl1B2B KERNEL(AffineMultiBiasKernelImpl1B2B)
#define RecurrentKernelImpl1B2B KERNEL(RecurrentKernelImpl1B2B)
#define DiagonalKernelImpl1B2B KERNEL(DiagonalKernelImpl1B2B)
#define TransposeKernelImpl2B KERNEL(TransposeKernelImpl2B)

#ifdef __cplusplus
extern "C" {
#endif

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
void AffineKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// uses active outputs list
void AffineActiveListKernelImpl1B(
    ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);

// Calculates affine transform on interleaved input vectors
// (input vectors in N columns, vector elements in K rows)
// handles multi bias
void AffineMultiBiasKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config);

// Calculates recurrent transform on flat input vectors
// (input vectors in N rows, vector elements in K columns)
void RecurrentKernelImpl1B(ExecutionKernelConfig<RecurrentConfig> const * const config);

void DiagonalKernelImpl1B(ExecutionKernelConfig<AffineConfig> const * const config);

void TransposeKernelImpl2B(TransposeConfig const * const transposeConfig);

#if OPT_LEVEL < 2
void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig);
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineActiveListKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config);
void RecurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void RecurrentKernelImpl1B2B(ExecutionKernelConfig<RecurrentConfig> const * const config);
void DiagonalKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void DiagonalKernelImpl1B2B(ExecutionKernelConfig<AffineConfig> const * const config);
#endif

#if OPT_LEVEL == 6 || OPT_LEVEL == 7
void TransposeKernelImpl1B(TransposeConfig const * const transposeConfig);
#endif

#if OPT_LEVEL == 7
void AffineKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineMultiBiasKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config);
void AffineActiveListKernelImpl1B1B(ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);
void RecurrentKernelImpl1B1B(ExecutionKernelConfig<RecurrentConfig> const * const config);
#endif

#ifdef __cplusplus
}
#endif
