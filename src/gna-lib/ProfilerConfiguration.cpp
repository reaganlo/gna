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

#include "ProfilerConfiguration.h"

#include "Expect.h"
#include "Request.h"

#include <cstdint>
#include <utility>

using namespace GNA;

const std::set<Gna2InstrumentationPoint>& ProfilerConfiguration::GetSupportedInstrumentationPoints()
{
    static const std::set<Gna2InstrumentationPoint> supportedInstrumentationPoints
    {
        Gna2InstrumentationPointLibPreprocessing,
        Gna2InstrumentationPointLibSubmission,
        Gna2InstrumentationPointLibProcessing,
        Gna2InstrumentationPointLibExecution,
        Gna2InstrumentationPointLibDeviceRequestReady,
        Gna2InstrumentationPointLibDeviceRequestSent,
        Gna2InstrumentationPointLibDeviceRequestCompleted,
        Gna2InstrumentationPointLibCompletion,
        Gna2InstrumentationPointLibReceived,
        Gna2InstrumentationPointDrvPreprocessing,
        Gna2InstrumentationPointDrvProcessing,
        Gna2InstrumentationPointDrvDeviceRequestCompleted,
        Gna2InstrumentationPointDrvCompletion,
        Gna2InstrumentationPointHwTotalCycles,
        Gna2InstrumentationPointHwStallCycles,
    };
    return supportedInstrumentationPoints;
}

const std::set<Gna2InstrumentationUnit>& ProfilerConfiguration::GetSupportedInstrumentationUnits()
{
    static const std::set<Gna2InstrumentationUnit> supportedInstrumentationUnits
    {
        Gna2InstrumentationUnitCycles,
        Gna2InstrumentationUnitMilliseconds,
        Gna2InstrumentationUnitMicroseconds,
    };
    return supportedInstrumentationUnits;
}

const std::set<Gna2InstrumentationMode>& ProfilerConfiguration::GetSupportedInstrumentationModes()
{
    static const std::set<Gna2InstrumentationMode> supportedInstrumentationModes
    {
        Gna2InstrumentationModeTotalStall,
        Gna2InstrumentationModeWaitForDmaCompletion,
        Gna2InstrumentationModeWaitForMmuTranslation,
        Gna2InstrumentationModeDescriptorFetchTime,
        Gna2InstrumentationModeInputBufferFillFromMemory,
        Gna2InstrumentationModeOutputBufferFullStall,
        Gna2InstrumentationModeOutputBufferWaitForIosfStall,
        Gna2InstrumentationModeDisabled,
    };
    return supportedInstrumentationModes;
}

uint32_t ProfilerConfiguration::GetMaxNumberOfInstrumentationPoints()
{
    return static_cast<uint32_t>(GetSupportedInstrumentationPoints().size());
}

ProfilerConfiguration::ProfilerConfiguration(const uint32_t configID,
    std::vector<Gna2InstrumentationPoint>&& selectedPoints,
    uint64_t* resultsIn) :
    ID{configID},
    Points{std::move(selectedPoints)},
    Results{ resultsIn }
{
    Expect::NotNull(Results);
    Expect::True(Points.size() <= GetMaxNumberOfInstrumentationPoints(), Gna2StatusIdentifierInvalid);
    auto leftToUse = GetSupportedInstrumentationPoints();
    for (const auto& selectedPoint: Points)
    {
        const auto numberOfErased = leftToUse.erase(selectedPoint);
        Expect::True(numberOfErased == 1, Gna2StatusDeviceParameterOutOfRange);
    }
}

void ProfilerConfiguration::SetUnit(Gna2InstrumentationUnit unitIn)
{
    Expect::True(GetSupportedInstrumentationUnits().count(unitIn) > 0, Gna2StatusIdentifierInvalid);
    Unit = unitIn;
}

Gna2InstrumentationUnit ProfilerConfiguration::GetUnit() const
{
    return Unit;
}

void ProfilerConfiguration::ExpectValid(Gna2InstrumentationMode encodingIn)
{
    Expect::True(GetSupportedInstrumentationModes().count(encodingIn) > 0, Gna2StatusIdentifierInvalid);
}

void ProfilerConfiguration::SetHwPerfEncoding(Gna2InstrumentationMode encodingIn)
{
    HwPerfEncoding = encodingIn;
}

uint8_t ProfilerConfiguration::GetHwPerfEncoding() const
{
    auto const encoding = static_cast<int>(HwPerfEncoding) + 1;
    return static_cast<uint8_t>(encoding & 0xFF);
}

void ProfilerConfiguration::SetResult(uint32_t const index, uint64_t const value) const
{
    Results[index] = value;
}
