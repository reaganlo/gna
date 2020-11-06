/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

struct GmmLayerCapabilities : LayerCapabilities
{
    static const FullCapabilitiesMap& GetOperands(uint32_t operandIndex);
};

/** Maximum number of mixture components per GMM State */
constexpr uint32_t GMM_MIXTURE_COMP_COUNT_MAX = 4096;

/** Maximum number of GMM states, active list elements and  */
constexpr uint32_t GMM_STATES_COUNT_MAX = 262144;

/** Size of memory alignment for mean, variance vectors and Gaussian Constants */
constexpr uint32_t GMM_MEM_ALIGNMENT = 8;

/** Mean vector width in bytes */
constexpr uint32_t GMM_MEAN_VALUE_SIZE = 1;

/** Minimum variance vector width in bytes */
constexpr uint32_t GMM_COVARIANCE_SIZE_MIN = 1;

/** Maximum variance vector width in bytes */
constexpr uint32_t GMM_COVARIANCE_SIZE_MAX = 2;

/** Score width in bytes */
constexpr uint32_t GMM_SCORE_SIZE = 4;

/** Minimum length of a vector */
constexpr uint32_t GMM_FV_ELEMENT_COUNT_MIN = 24;

/** The allowed alignment of vector lengths */
constexpr uint32_t GMM_FV_ELEMENT_COUNT_MULTIPLE_OF = 8;

/** Feature vector width in bytes */
constexpr uint32_t GMM_FV_ELEMENT_SIZE = 1;

/** Maximum length of a vector */
constexpr uint32_t GMM_FV_ELEMENT_COUNT_MAX = 96;

/** Gaussian Constants width in bytes */
constexpr uint32_t GMM_CONSTANTS_SIZE = 4;

}
