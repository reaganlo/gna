/*
 @copyright

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-instrumentation-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA_API_INSTRUMENTATION Instrumentation API

 API for querying inference performance statistics.

 @{
 *****************************************************************************/

#ifndef __GNA2_INSTRUMENTATION_API_H
#define __GNA2_INSTRUMENTATION_API_H

#include "gna2-common-api.h"

#include <stdint.h>

enum GnaInstrumentationPoint;
enum GnaInstrumentationUnit;
enum GnaInstrumentationMode;

/**
 Enables and configures instrumentation configuration.

 Instrumentation configurations have to be declared a priori to minimize the
 request preparation time and reduce processing latency.
 Configurations can be shared with multiple request configurations when the request
 with the current configuration has been completed and retrieved by GnaRequestWait().

 @see
    GnaRequestConfigSetInstrumentationUnit and ::GnaInstrumentationUnitMicroseconds
    for description of result units.

 @see GnaRequestConfigSetInstrumentationMode and GnaInstrumentationMode
    for description of hardware instrumentation.

 @param numberOfInstrumentationPoints A number of selected instrumentation points.
 @param selectedInstrumentationPoints An array of selected instrumentation points.
 @param results Buffer to save instrumentation results to.
    Result buffer size have to be at least numberOfInstrumentationPoints * sizeof(uint64_t).
 @param [out] instrumentationConfigId Identifier of created instrumentation configuration.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaInstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    enum GnaInstrumentationPoint* selectedInstrumentationPoints,
    uint64_t * results,
    uint32_t * instrumentationConfigId);

/**
 Inference request instrumentation points.
 */
enum GnaInstrumentationPoint
{
    /**
     Request preprocessing start, from library instrumentation.
     */
    GnaInstrumentationPointLibPreprocessing = 0,

    /**
     Request submission start, from library instrumentation.
     */
    GnaInstrumentationPointLibSubmission = 1,

    /**
     Request processing start, from library instrumentation.

     */
    GnaInstrumentationPointLibProcessing = 2,

    /**
     Request execution start, from library instrumentation.
     Actual software computation or issuing device request.
     */
    GnaInstrumentationPointLibExecution = 3,

    /**
     Request ready to send to device, from library instrumentation.
     */
    GnaInstrumentationPointLibDeviceRequestReady = 4,

    /**
     Request ready to send to device, from library instrumentation.
     */
    GnaInstrumentationPointLibDeviceRequestSent = 5,

    /**
     Request completed by device, from library instrumentation.
     */
    GnaInstrumentationPointLibDeviceRequestComepleted = 6,

    /**
     Request execution completed, from library instrumentation.
     Actual software computation done or device request notified.
     */
    GnaInstrumentationPointLibCompletion = 7,

    /**
     Request received by user, from library instrumentation.
     */
    GnaInstrumentationPointLibReceived = 8,

    /**
     Request preprocessing start, from driver instrumentation.
     */
    GnaInstrumentationPointDrvPreprocessing = 9,

    /**
     Request processing started by hardware, from driver instrumentation.
     */
    GnaInstrumentationPointDrvProcessing = 10,

    /**
     Request completed interrupt triggered by hardware, from driver instrumentation.
     */
    GnaInstrumentationPointDrvDeviceRequestComepleted = 11,

    /**
     Request execution completed, from driver instrumentation.
     Driver completed interrupt and request handling.
     */
    GnaInstrumentationPointDrvCompletion = 12,

    /**
     Total time spent on processing in hardware.
     Total = Compute + Stall
     @warning This event always provides time duration instead of time point.
     */
    GnaInstrumentationPointHwTotalCycles = 13,

    /**
     Time hardware spent on waiting for data.
     @warning This event always provides time duration instead of time point.
     */
    GnaInstrumentationPointHwStallCycles = 14,
};

/**
 Assigns instrumentation config to given request configuration.

 @see GnaRequestConfigRelease()

 @param instrumentationConfigId Identifier of instrumentation config used.
 @param requestConfigId Request configuration to modify.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaInstrumentationConfigAssignToRequestConfig(
    uint32_t instrumentationConfigId,
    uint32_t requestConfigId);

/**
 Sets instrumentation unit for given configuration.

 Instrumentation results will represent a value in selected units.
 @note
    ::GnaInstrumentationUnitMicroseconds is used when not set.

 @param instrumentationConfigId Instrumentation configuration to modify.
 @param instrumentationUnit Type of hardware performance statistic.
 */
GNA_API enum GnaStatus GnaInstrumentationConfigSetUnit(
    uint32_t instrumentationConfigId,
    enum GnaInstrumentationUnit instrumentationUnit);

/**
 Units that instrumentation will count and report.
 */
enum GnaInstrumentationUnit
{
    /**
     Microseconds.

     Uses std::chrono. @see http://www.cplusplus.com/reference/chrono/
     */
    GnaInstrumentationUnitMicroseconds = GNA_DEFAULT,

    /**
     Milliseconds.

     Uses std::chrono. @see http://www.cplusplus.com/reference/chrono/
     */
    GnaInstrumentationUnitMilliseconds = 1,

    /**
     Processor cycles.

     Uses RDTSC. @see https://en.wikipedia.org/wiki/Time_Stamp_Counter
     */
    GnaInstrumentationUnitCycles = 2,
};

/**
 Sets hardware instrumentation mode for given configuration.

 @note
    ::GnaInstrumentationModeTotalStall is used when not set.

 @param instrumentationConfigId Instrumentation configuration to modify.
 @param instrumentationMode Mode of hardware instrumentation.
 */
GNA_API enum GnaStatus GnaInstrumentationConfigSetMode(
    uint32_t instrumentationConfigId,
    enum GnaInstrumentationMode instrumentationMode);

/**
 Mode of instrumentation for hardware performance counters.

 When performance counting is enabled, the total scoring cycles counter is always on.
 In addition one of several reasons for stall may be measured to allow
 identifying the bottlenecks in the scoring operation.
 */
enum GnaInstrumentationMode
{
    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeTotalStall = GNA_DEFAULT,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeWaitForDmaCompletion = 1,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeWaitForMmuTranslation = 2,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeDescriptorFetchTime = 3,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeInputBufferFillFromMemory = 4,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeOutputBufferFullStall = 5,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeOutputBufferWaitForIosfStall = 6,

    /**
     TODO:3:API: add comment
     */
    GnaInstrumentationModeDisabled = GNA_DISABLED,
};

/**
 Releases instrumentation config and its resources.

 @note Please make sure all requests using this config are completed.

 @param instrumentationConfigId Identifier of affected instrumentation configuration.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaInstrumentationConfigRelease(
    uint32_t instrumentationConfigId);

#endif // __GNA2_INSTRUMENTATION_API_H

/**
 @}
 @}
 */
