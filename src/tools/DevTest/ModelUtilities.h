
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

#include "gna-api.h"

#include <cstdint>
#include <stddef.h>

class ModelUtilities
{
public:
    static size_t CalculateDnnSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                   uint32_t bytesPerWeight, uint32_t nSegments);

    static size_t CalculateMultibiasSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                   uint32_t bytesPerWeight, uint32_t nSegments);

    static size_t CalculateRnnSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                   uint32_t bytesPerWeight, uint32_t nSegments);

    static size_t CalculateCnnSize(uint32_t inputElementCount, uint32_t outputsPerFilter,
                                   uint32_t nFilters, uint32_t nFilterCoeficcients, uint32_t nSegments);

    static size_t CalculateSimpleSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount);

    static size_t CalculateGmmSize(uint32_t vectorCount, uint32_t stateCount, uint32_t mixtureCount, uint32_t inputElementCount, gna_gmm_mode gmmMode);

    static void GeneratePwlSegments(intel_pwl_segment_t *segments, uint32_t nSegments);

};
