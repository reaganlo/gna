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

#include "Capabilities.h"
#include "DataMode.h"

#include <vector>

namespace GNA
{

using ComponentFullCapabilityMap = std::map<const uint32_t, FullCapabilitiesMap>;

struct LayerCapabilities
{
    /** Number of input groups constraint - max */
    static constexpr uint32_t BatchSizeMax = 8;

    /** Number of input groups constraint - max */
    static constexpr uint32_t InputGroupsCountMax = 8;

    /** Total number of input elements constraint - must be multiple of */
    static constexpr uint32_t InputElementsMultipllier = 8;

    /** Number of input groups constraint for Copy layer 3.0- max */
    static constexpr uint32_t CopyRowsMax = 255;

    /** Total number of input elements constraint - must be multiple of */
    static constexpr uint32_t InputElementCountMultiplier = 8;

    /** Total number of output elements constraint - must be multiple of */
    static constexpr uint32_t RecurrentOutputElementCountMultiplier = 32;

    /** Total number of input elements constraint - max elements */
    static constexpr uint32_t InputElementCountMax = UINT16_MAX;

    /** Number of pwl segments constraint - max  */
    static constexpr uint32_t ActivationFunctionSegmentCountMax = 128;

    /** Number of pwl segments constraint - min  */
    static constexpr uint32_t ActivationFunctionSegmentCountMin = 2;

    /** Weight elements size constraint - max size B */
    static constexpr uint32_t WeightElementSizeMax = 2;

    static const MultiplierMap & InputElementCountMultipliers();

    static const DataModeLimits & GetModes(uint32_t operandIndex, gna_device_generation generation);

    static const RangeLimits<>& limitsForInput();

    static const RangeLimits<>& limitsForOutput();

    static const RangeLimits<>& limitsForInputShapeLegacy();

    static const RangeLimits<>& limitsForOutputShapeLegacy();

    static const RangeLimits<>& limitsForInputGroupsMax();

    static const RangeLimits<>& limitsForOutputGroupsMax();
};

}
