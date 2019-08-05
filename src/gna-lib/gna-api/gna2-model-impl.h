/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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
#ifndef __GNA2_MODEL_IMPL_H
#define __GNA2_MODEL_IMPL_H

#include "gna2-model-api.h"

namespace GNA
{

typedef struct Gna2Model ApiModel;
typedef struct Gna2Operation ApiOperation;
typedef struct Gna2Shape ApiShape;
typedef struct Gna2Tensor ApiTensor;
typedef struct Gna2ModelError ModelError;

typedef enum Gna2OperationType OperationType;
typedef enum Gna2TensorMode TensorMode;
typedef enum Gna2DataType DataType;
typedef enum Gna2BiasMode ApiBiasMode;
typedef enum Gna2PoolingMode Mode;
typedef enum Gna2ErrorType ErrorType;
typedef enum Gna2ItemType ItemType;

constexpr uint32_t ScratchpadOperandIndex = UINT32_MAX;
constexpr uint32_t InputOperandIndex = 0;
constexpr uint32_t OutputOperandIndex = 1;
constexpr uint32_t WeightOperandIndex = 2;
constexpr uint32_t FilterOperandIndex = 2;
constexpr uint32_t BiasOperandIndex = 3;
constexpr uint32_t PwlOperandIndex = 4;
constexpr uint32_t WeightScaleFactorOperandIndex = 5;
constexpr uint32_t GmmInterleavedOperandIndex = 2;
constexpr uint32_t GmmMeanOperandIndex = 2;
constexpr uint32_t GmmInverseCovarianceOperandIndex = 3;
constexpr uint32_t GmmGaussianConstantOperandIndex = 4;

// NOTE: temporary solution for simple and fast kernel buffer indexing, always set as last + 1 operand index 
constexpr uint32_t ScratchpadOperandKernelIndex = 6;

}

#endif // __GNA2_MODEL_IMPL_H
