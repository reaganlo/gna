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

#include "DriverInterface.h"

using namespace GNA;

const DriverCapabilities& DriverInterface::GetCapabilities() const
{
    return driverCapabilities;
}

void DriverInterface::convertPerfResultUnit(DriverPerfResults & driverPerf,
    Gna2InstrumentationUnit targetUnit) const
{
    auto const frequency = driverCapabilities.perfCounterFrequency;

    switch (targetUnit)
    {
    case Gna2InstrumentationUnitMicroseconds:
        return convertPerfResultUnit(driverPerf, frequency, RequestProfiler::MICROSECOND_MULTIPLIER);
    case Gna2InstrumentationUnitMilliseconds:
        return convertPerfResultUnit(driverPerf, frequency, RequestProfiler::MILLISECOND_MULTIPLIER);
    default:
        // no conversion required
        break;
    }
}

void DriverInterface::convertPerfResultUnit(DriverPerfResults& driverPerf,
    uint64_t frequency, uint64_t multiplier)
{
    if (0 == frequency || 0 == multiplier)
    {
        throw GnaException(Gna2StatusNullArgumentNotAllowed);
    }
    auto const newProcessing = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        driverPerf.Preprocessing, driverPerf.Processing);
    driverPerf.Preprocessing = 0;

    auto const newRequestCompleted = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        driverPerf.Processing, driverPerf.DeviceRequestCompleted);
    driverPerf.Processing = newProcessing;

    auto const newRequestCompletion = RequestProfiler::ConvertElapsedTime(frequency, multiplier,
        driverPerf.DeviceRequestCompleted, driverPerf.Completion);
    driverPerf.DeviceRequestCompleted = newProcessing + newRequestCompleted;
    driverPerf.Completion = newProcessing + newRequestCompleted + newRequestCompletion;
}

