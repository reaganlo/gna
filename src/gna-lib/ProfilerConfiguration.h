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
#include "gna2-instrumentation-api.h"

#include <set>

namespace GNA
{
class ProfilerConfiguration
{
public:

    static uint32_t GetMaxNumberOfInstrumentationPoints();
    static const std::set<Gna2InstrumentationPoint>& GetSupportedInstrumentationPoints();
    static const std::set<Gna2InstrumentationUnit>& GetSupportedInstrumentationUnits();
    static const std::set<Gna2InstrumentationMode>& GetSupportedInstrumentationModes();
    static void ExpectValid(Gna2InstrumentationMode encodingIn);

    ProfilerConfiguration(uint32_t configID,
        uint32_t numberOfPoints,
        const Gna2InstrumentationPoint* selectedPoints,
        uint64_t* selectedResults);

    const uint32_t ID;
    const Gna2InstrumentationPoint* const Points;
    const uint32_t NPoints;

    void SetUnit(Gna2InstrumentationUnit unitIn);
    Gna2InstrumentationUnit GetUnit() const;

    void SetHwPerfEncoding(Gna2InstrumentationMode encodingIn);
    uint8_t GetHwPerfEncoding() const;

    void SetResult(uint32_t index, uint64_t value);

private:
    uint64_t* const Results;

    Gna2InstrumentationMode HwPerfEncoding = Gna2InstrumentationModeTotalStall;
    Gna2InstrumentationUnit Unit = Gna2InstrumentationUnitMicroseconds;
};

}
