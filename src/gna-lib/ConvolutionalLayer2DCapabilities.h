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

#pragma once

#include "LayerCapabilities.h"

namespace GNA
{

struct ConvolutionalLayer2DCapabilities : LayerCapabilities
{
    static const FullCapabilitiesMap & GetOperands(uint32_t operandIndex);
    static const FullCapabilitiesMap & GetParameters(uint32_t parameterIndex);

    static const OperationCapabilityMap & GetOperands(uint32_t operandIndex, nn_operation operation);
    static const OperationCapabilityMap & GetParameters(uint32_t parameterIndex, nn_operation operation);

    /** CNN minimum number of filter coefficients */
    static constexpr uint32_t Filter1DElementsMin = 8;

    /** CNN maximum number of filter coefficients */
    static constexpr uint32_t Filter1DElementsMax = 768;

    /** CNN 2D minimum number of kernel elements in one dimension */
    static constexpr uint32_t Filter2DElementsMin = 1;

    /** CNN 2D maximum number of kernel elements in one dimension */
    static constexpr uint32_t Filter2DElementsMax = 255;

    /** CNN number of filter coefficients constraint - must be multiple of */
    static constexpr uint32_t Filter1DElementsMultiplier = 4;

    /** CNN maximum number of filters */
    static constexpr uint32_t Filter1DCountMax = ((UINT16_MAX + 1) - 4);

    /** CNN 2D maximum number of kernels */
    static constexpr uint32_t Filter2DCountMax = 8192;

    /** CNN 2D maximum kernel depth */
    static constexpr uint32_t Filter2DDepthMax = 2048;

    /** CNN minimum size of pooling window */
    static constexpr uint32_t PoolingWindowSizeMin = 1;

    /** CNN maximum size of pooling window */
    static constexpr uint32_t PoolingWindowSizeMax = 6;

    /** CNN 1D maximum number of kernel elements in one dimension For int8_t */
    static constexpr uint32_t Kernel1DElementsPerDimensionMax = 2048;

};

}
