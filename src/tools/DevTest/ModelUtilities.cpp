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

#include "ModelUtilities.h"

size_t ModelUtilities::CalculateDnnSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                        uint32_t bytesPerWeight, uint32_t nSegments)
{
    size_t inputBytes = ALIGN64(vectorCount * inputElementCount * sizeof(int16_t));
    size_t weightBytes = ALIGN64(outputElementCount * inputElementCount * bytesPerWeight);
    size_t biasBytes = outputElementCount * ((bytesPerWeight == sizeof(int32_t))
                        ? GNA_INT32: GNA_DATA_RICH_FORMAT);
    size_t outputBytes = 0;
    size_t tmpOutputBytes = 0;
    size_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int16_t));
        tmpOutputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int32_t));
        pwlBytes = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));
    }
    else
    {
        outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int32_t));
    }

    return inputBytes + outputBytes + tmpOutputBytes + biasBytes + weightBytes + pwlBytes;
}

size_t ModelUtilities::CalculateMultibiasSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                              uint32_t bytesPerWeight, uint32_t nSegments)
{
    size_t inputBytes = ALIGN64(vectorCount * inputElementCount * sizeof(int16_t));
    size_t weightBytes = ALIGN64(outputElementCount * inputElementCount * bytesPerWeight);
    size_t biasBytes = vectorCount * outputElementCount * ((bytesPerWeight == sizeof(int32_t))
                        ? GNA_INT32: GNA_DATA_RICH_FORMAT);
    size_t scaleBytes = (sizeof(int16_t) == bytesPerWeight) ? 0 : outputElementCount * sizeof(intel_compound_bias_t);
    size_t outputBytes = 0;
    size_t tmpOutputBytes = 0;
    size_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int16_t));
        tmpOutputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int32_t));
        pwlBytes = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));
    }
    else
    {
        outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int32_t));
    }

    return inputBytes + outputBytes + tmpOutputBytes + biasBytes + scaleBytes + weightBytes + pwlBytes;
}

size_t ModelUtilities::CalculateCnnSize(uint32_t inputElementCount, uint32_t outputsPerFilter,
                                        uint32_t nFilters, uint32_t nFilterCoeficcients, uint32_t nSegments)
{
    size_t inputBytes = ALIGN64(inputElementCount * sizeof(int16_t));
    size_t filterBytes = ALIGN64(nFilters * nFilterCoeficcients * sizeof(int16_t));
    size_t biasBytes = ALIGN64(outputsPerFilter * sizeof(int32_t));
    size_t outputBytes = 0;
    size_t tmpOutputBytes = 0;
    size_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = ALIGN64(outputsPerFilter * nFilters * sizeof(int16_t));
        tmpOutputBytes = ALIGN64(outputsPerFilter * nFilters * sizeof(int32_t));
        pwlBytes = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));
    }
    else
    {
        outputBytes = ALIGN64(outputsPerFilter * nFilters * sizeof(int32_t));
    }

    return inputBytes + filterBytes + outputBytes + tmpOutputBytes + biasBytes + pwlBytes;
}

size_t ModelUtilities::CalculateRnnSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                        uint32_t bytesPerWeight, uint32_t nSegments)
{
    size_t inputBytes = ALIGN64(vectorCount * inputElementCount * sizeof(int16_t));
    size_t weightBytes = ALIGN64((inputElementCount + outputElementCount) * outputElementCount * bytesPerWeight);
    size_t biasBytes = outputElementCount * ((bytesPerWeight == sizeof(int32_t))
                        ? GNA_INT32: GNA_DATA_RICH_FORMAT);
    size_t outputBytes = 0;
    size_t tmpOutputBytes = 0;
    size_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int16_t));
        tmpOutputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int32_t));
        pwlBytes = ALIGN64(nSegments * sizeof(intel_pwl_segment_t));
    }
    else
    {
        outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int32_t));
    }

    return inputBytes + outputBytes + tmpOutputBytes + biasBytes + weightBytes + pwlBytes;
}

size_t ModelUtilities::CalculateSimpleSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount)
{
    size_t inputBytes = ALIGN64(vectorCount * inputElementCount * sizeof(int16_t));
    size_t outputBytes = ALIGN64(vectorCount * outputElementCount * sizeof(int16_t));

    return inputBytes + outputBytes;
}

size_t ModelUtilities::CalculateGmmSize(uint32_t vectorCount, uint32_t stateCount, uint32_t mixtureCount, uint32_t inputElementCount, gna_gmm_mode gmmMode)
{
    size_t varsBytes   = 0;
    if(GNA_MAXMIX16 == gmmMode)
    {
        varsBytes = ALIGN64(stateCount * mixtureCount * inputElementCount * sizeof(uint16_t));
    }
    else
    {
        varsBytes = ALIGN64(stateCount * mixtureCount * inputElementCount * sizeof(uint8_t));
    }

    size_t meanBytes   = ALIGN64(stateCount * mixtureCount * inputElementCount * sizeof(uint8_t));
    size_t constBytes  = ALIGN64(stateCount * mixtureCount * sizeof(uint32_t));
    size_t inputBytes  = ALIGN64(vectorCount * inputElementCount * sizeof(uint8_t));
    size_t outputBytes = ALIGN64(vectorCount * stateCount * sizeof(int32_t));       // (4 out vectors, 8 elems in each one, 4-byte elems)
    size_t scratchpadBytes = outputBytes;

    return varsBytes + meanBytes + constBytes + inputBytes + outputBytes + scratchpadBytes;
}

void ModelUtilities::GeneratePwlSegments(intel_pwl_segment_t *segments, uint32_t nSegments)
{
    auto xBase = INT32_MIN;
    auto xBaseInc = UINT32_MAX / nSegments;
    auto yBase = INT32_MAX;
    auto yBaseInc = UINT16_MAX / nSegments;
    for (auto i = uint32_t{0}; i < nSegments; i++, xBase += xBaseInc, yBase += yBaseInc)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = static_cast<int16_t>(yBase);
        segments[i].slope = static_cast<int16_t>(1);
    }
}
