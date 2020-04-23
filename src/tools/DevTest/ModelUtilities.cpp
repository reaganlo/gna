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

#include "ModelUtilities.h"

#include <cstdint>

uint32_t ModelUtilities::CalculateDnnSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                        uint32_t bytesPerWeight, uint32_t nSegments)
{
    const auto inputBytes = CastAndRoundUpTo64(vectorCount * inputElementCount * sizeof(int16_t));
    const auto weightBytes = CastAndRoundUpTo64(outputElementCount * inputElementCount * bytesPerWeight);
    const auto biasBytes = static_cast<uint32_t>(outputElementCount * ((bytesPerWeight == sizeof(int16_t))
                        ? sizeof(int32_t): sizeof(Gna2CompoundBias)));
    uint32_t outputBytes = 0;
    uint32_t tmpOutputBytes = 0;
    uint32_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int16_t));
        tmpOutputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int32_t));
        pwlBytes = CastAndRoundUpTo64(nSegments * sizeof(Gna2PwlSegment));
    }
    else
    {
        outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int32_t));
    }

    return inputBytes + outputBytes + tmpOutputBytes + biasBytes + weightBytes + pwlBytes;
}

uint32_t ModelUtilities::CalculateMultibiasSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                              uint32_t bytesPerWeight, uint32_t nSegments)
{
    const auto inputBytes = CastAndRoundUpTo64(vectorCount * inputElementCount * sizeof(int16_t));
    const auto weightBytes = CastAndRoundUpTo64(outputElementCount * inputElementCount * bytesPerWeight);
    const auto biasBytes = static_cast<uint32_t>(vectorCount * outputElementCount * sizeof(int32_t));

    const auto scaleBytes = static_cast<uint32_t>(sizeof(int16_t) == bytesPerWeight ?
                    0 : outputElementCount * sizeof(Gna2CompoundBias));
    uint32_t outputBytes;
    uint32_t tmpOutputBytes = 0;
    uint32_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int16_t));
        tmpOutputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int32_t));
        pwlBytes = CastAndRoundUpTo64(nSegments * sizeof(Gna2PwlSegment));
    }
    else
    {
        outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int32_t));
    }

    return inputBytes + outputBytes + tmpOutputBytes + biasBytes + scaleBytes + weightBytes + pwlBytes;
}

uint32_t ModelUtilities::CalculateCnnSize(uint32_t inputElementCount, uint32_t outputsPerFilter,
                                        uint32_t nFilters, uint32_t nFilterCoeficcients, uint32_t nSegments)
{
    const auto inputBytes = CastAndRoundUpTo64(inputElementCount * sizeof(int16_t));
    const auto filterBytes = CastAndRoundUpTo64(nFilters * nFilterCoeficcients * sizeof(int16_t));
    const auto biasBytes = CastAndRoundUpTo64(outputsPerFilter * sizeof(int32_t));
    uint32_t outputBytes = 0;
    uint32_t tmpOutputBytes = 0;
    uint32_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = CastAndRoundUpTo64(outputsPerFilter * nFilters * sizeof(int16_t));
        tmpOutputBytes = CastAndRoundUpTo64(outputsPerFilter * nFilters * sizeof(int32_t));
        pwlBytes = CastAndRoundUpTo64(nSegments * sizeof(Gna2PwlSegment));
    }
    else
    {
        outputBytes = CastAndRoundUpTo64(outputsPerFilter * nFilters * sizeof(int32_t));
    }

    return inputBytes + filterBytes + outputBytes + tmpOutputBytes + biasBytes + pwlBytes;
}

uint32_t ModelUtilities::CalculateRnnSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount,
                                        uint32_t bytesPerWeight, uint32_t nSegments)
{
    const auto inputBytes = CastAndRoundUpTo64(vectorCount * inputElementCount * sizeof(int16_t));
    const auto weightBytes = CastAndRoundUpTo64((inputElementCount + outputElementCount) * outputElementCount * bytesPerWeight);
    const auto biasBytes = static_cast<uint32_t>(outputElementCount * ((bytesPerWeight == sizeof(int16_t))
                        ? sizeof(int16_t): sizeof(Gna2CompoundBias)));
    uint32_t outputBytes;
    uint32_t tmpOutputBytes = 0;
    uint32_t pwlBytes = 0;

    if (nSegments > 0)
    {
        outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int16_t));
        tmpOutputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int32_t));
        pwlBytes = CastAndRoundUpTo64(nSegments * sizeof(Gna2PwlSegment));
    }
    else
    {
        outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int32_t));
    }

    return inputBytes + outputBytes + tmpOutputBytes + biasBytes + weightBytes + pwlBytes;
}

uint32_t ModelUtilities::CalculateSimpleSize(uint32_t vectorCount, uint32_t inputElementCount, uint32_t outputElementCount)
{
    const auto inputBytes = CastAndRoundUpTo64(vectorCount * inputElementCount * sizeof(int16_t));
    const auto outputBytes = CastAndRoundUpTo64(vectorCount * outputElementCount * sizeof(int16_t));

    return inputBytes + outputBytes;
}

uint32_t ModelUtilities::CalculateGmmSize(uint32_t vectorCount, uint32_t stateCount, uint32_t mixtureCount, uint32_t inputElementCount, const uint32_t bytesPerMaxMix)
{
    const auto varsBytes = CastAndRoundUpTo64(stateCount * mixtureCount * inputElementCount * bytesPerMaxMix);


    const auto meanBytes   = CastAndRoundUpTo64(stateCount * mixtureCount * inputElementCount * sizeof(uint8_t));
    const auto constBytes  = CastAndRoundUpTo64(stateCount * mixtureCount * sizeof(uint32_t));
    const auto inputBytes  = CastAndRoundUpTo64(vectorCount * inputElementCount * sizeof(uint8_t));
    const auto outputBytes = CastAndRoundUpTo64(vectorCount * stateCount * sizeof(int32_t));       // (4 out vectors, 8 elems in each one, 4-byte elems)
    const auto scratchpadBytes = outputBytes;

    return varsBytes + meanBytes + constBytes + inputBytes + outputBytes + scratchpadBytes;
}

void ModelUtilities::GeneratePwlSegments(Gna2PwlSegment* segments, uint32_t nSegments)
{
    auto xBase = INT32_MIN;
    auto yBase = INT32_MAX;
    const auto xBaseInc = UINT32_MAX / nSegments;
    const auto yBaseInc = UINT16_MAX / nSegments;
    for (auto i = uint32_t{ 0 }; i < nSegments; i++, xBase += xBaseInc, yBase += yBaseInc)
    {
        segments[i].xBase = xBase;
        segments[i].yBase = static_cast<int16_t>(yBase);
        segments[i].Slope = static_cast<int16_t>(1);
    }
}
