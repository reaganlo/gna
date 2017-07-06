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

#include <minmax.h>

#include "Logger.h"

using namespace GNA;

const char* const Logger::StatusStrings[] =
{
    "GNA_SUCCESS", " - Success: Operation successful, no errors or warnings",
    "GNA_DEVICEBUSY", " - Warning: Device busy - accelerator is still running, can not enqueue more requests",
    "GNA_SSATURATE", " - Warning: Scoring saturation - an arithmetic operation has resulted in saturation",
    "GNA_UNKNOWN_ERROR", " - Unknown error occurred",
    "GNA_ERR_QUEUE", " - Queue can not create or enqueue more requests",
    "GNA_READFAULT", " - Scoring data: invalid input",
    "GNA_WRITEFAULT", " - Scoring data: invalid output buffer",
    "GNA_BADFEATWIDTH", " - Feature vector: width not supported",
    "GNA_BADFEATLENGTH", " - Feature vector: length not supported",
    "GNA_BADFEATOFFSET", " - Feature vector: offset not supported",
    "GNA_BADFEATALIGN", " - Feature vector: invalid memory alignment",
    "GNA_BADFEATNUM", " - Feature vector: Number of feature vectors not supported",

    "GNA_INVALIDINDICES", " - Scoring data: number of active indices  not supported",
    "GNA_DEVNOTFOUND", " - Device: device not available",
    "GNA_OPENFAILURE", " - Device: internal error occurred during opening device",
    "GNA_INVALIDHANDLE", " - Device: invalid handle",
    "GNA_CPUTYPENOTSUPPORTED", " - Device: processor type not supported",
    "GNA_PARAMETEROUTOFRANGE", " - Device: GMM Parameter out of Range error occurred",
    "GNA_VAOUTOFRANGE", " - Device: Virtual Address out of range on DMA ch.",
    "GNA_UNEXPCOMPL", " - Device: Unexpected completion during PCIe operation",
    "GNA_DMAREQERR", " - Device: DMA error during PCIe operation",
    "GNA_MMUREQERR", " - Device: MMU error during PCIe operation",
    "GNA_BREAKPOINTPAUSE", " - Device: GMM accelerator paused on breakpoint",
    "GNA_BADMEMALIGN", " - Device: invalid memory alignment",
    "GNA_INVALIDMEMSIZE", " - Device: requested memory size not supported",
    "GNA_MODELSIZEEXCEEDED", " - Device: request's model configuration exceeded supported GNA_HW mode limits",
    "GNA_BADREQID", " - Device: invalid scoring request identifier",
    "GNA_WAITFAULT", " - Device: wait failed",
    "GNA_IOCTLRESERR", " - Device: IOCTL result retrieval failed",
    "GNA_IOCTLSENDERR", " - Device: sending IOCTL failed",
    "GNA_NULLARGNOTALLOWED", " - NULL argument not allowed",
    "GNA_INVALID_MODEL", " - Given model is invalid",
    "GNA_INVALID_REQUEST_CONFIGURATION", " - Given request configuration is invalid",
    "GNA_NULLARGREQUIRED", " - NULL argument is required",
    "GNA_ERR_MEM_ALLOC1", " - Memory: Already allocated, only single allocation per device is allowed",
    "GNA_ERR_RESOURCES", " - Unable to create new resources",
    "GNA_ERR_NOT_MULTIPLE", " - Value is not multiple of required value",
    "GNA_ERR_DEV_FAILURE", " - Critical device error occurred, device has been reset",

    "GMM_BADMEANWIDTH", " - Mean vector: width not supported",
    "GMM_BADMEANOFFSET", " - Mean vector: offset not supported",
    "GMM_BADMEANSETOFF", " - Mean vector: set offset not supported",
    "GMM_BADMEANALIGN", " - Mean vector: invalid memory alignment",
    "GMM_BADVARWIDTH", " - Variance vector: width not supported",
    "GMM_BADVAROFFSET", " - Variance vector: offset not supported",
    "GMM_BADVARSETOFF", " - Variance vector: set offset not supported",
    "GMM_BADVARSALIGN", " - Variance vector: invalid memory alignment",
    "GMM_BADGCONSTOFFSET", " - Gconst: set offset not supported",
    "GMM_BADGCONSTALIGN", " - Gconst: invalid memory alignment",
    "GMM_BADMIXCNUM", " - Scoring data: number of mixture components not supported",
    "GMM_BADNUMGMM", " - Scoring data: number of GMMs not supported",
    "GMM_BADMODE", " - Scoring data: GMM scoring mode not supported",
    "GMM_CFG_INVALID_LAYOUT", " - GMM Data layout is invalid",

    "XNN_ERR_NET_LYR_NO", " - XNN: Not supported number of layers",
    "XNN_ERR_NETWORK_INPUTS", " - XNN: Network is invalid - input buffers number differs from input layers number",
    "XNN_ERR_NETWORK_OUTPUTS", " - XNN: Network is invalid - output buffers number differs from output layers number",
    "XNN_ERR_LYR_KIND", " - XNN: Not supported layer kind",
    "XNN_ERR_LYR_TYPE", " - XNN: Not supported layer type",
    "XNN_ERR_LYR_CFG", " - XNN: Invalid layer configuration",
    "XNN_ERR_NO_FEEDBACK", " - XNN: No RNN feedback buffer specified",
    "XNN_ERR_NO_LAYERS", " - XNN: At least one layer must be specified",
    "XNN_ERR_GROUPING", " - XNN: Invalid grouping factor",
    "XNN_ERR_INPUT_BYTES", " - XNN: Invalid number of bytes per input",
    "XNN_ERR_INT_OUTPUT_BYTES", " - XNN: Invalid number of bytes per intermediate output",
    "XNN_ERR_OUTPUT_BYTES", " - XNN: Invalid number of bytes per output",
    "XNN_ERR_WEIGHT_BYTES", " - XNN: Invalid number of bytes per weight",
    "XNN_ERR_BIAS_BYTES", " - XNN: Invalid number of bytes per bias",
    "XNN_ERR_BIAS_MULTIPLIER", " - XNN: Multiplier larger than 255",
    "XNN_ERR_BIAS_INDEX", " - XNN: Bias Vector index larger than grouping factor",
    "XNN_ERR_PWL_SEGMENTS", " - XNN: Activation function segment number is invalid",
    "XNN_ERR_PWL_DATA", " - XNN: Activation function enabled but segment data not set",
    "XNN_ERR_MM_INVALID_IN", " - XNN: Invalid input data or configuration in matrix mul. op.",
    "CNN_ERR_FLT_COUNT", " - CNN Layer: invalid number of filters",
    "CNN_ERR_FLT_STRIDE", " - CNN Layer: invalid filter stride",
    "CNN_ERR_POOL_STRIDE", " - CNN Layer: invalid pool stride",

    "UNKNOWN STATUS", " - Status code is invalid"
};

Logger::Logger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
    const char * const levelErrorIn) :
    defaultStream{defaultStreamIn},
    component{componentIn},
    levelMessage{levelMessageIn},
    levelError{levelErrorIn}
{}

void Logger::LineBreak() const
{}

void Logger::HorizontalSpacer() const
{}

void Logger::Message(const status_t status) const
{
    UNREFERENCED_PARAMETER(status);
}

void Logger::Message(const status_t status, const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(status);
    UNREFERENCED_PARAMETER(format);
}

void Logger::Message(const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(format);
}

void Logger::Error(const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(format);
}

void Logger::Error(const status_t status, const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(status);
    UNREFERENCED_PARAMETER(format);
}

void Logger::Error(const status_t status) const
{
    UNREFERENCED_PARAMETER(status);
}

const char * const Logger::StatusToString(const intel_gna_status_t status) noexcept
{
    const auto statusSafe = min(status, NUMGNASTATUS);
    return StatusStrings[2 * statusSafe];
}

DebugLogger::DebugLogger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
    const char * const levelErrorIn) :
    Logger(defaultStreamIn, componentIn, levelMessageIn, levelErrorIn)
{}

void DebugLogger::LineBreak() const
{
    fprintf(defaultStream, "\n");
}

void DebugLogger::Message(const status_t status) const
{
    printMessage(&status, nullptr, nullptr);
}

void DebugLogger::Message(const status_t status, const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printMessage(&status, format, args);
    va_end(args);
}

void DebugLogger::Message(const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printMessage(nullptr, format, args);
    va_end(args);
}

void DebugLogger::Error(const status_t status) const
{
    printError(&status, nullptr, nullptr);
}

void DebugLogger::Error(const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printError(nullptr, format, args);
    va_end(args);
}

void DebugLogger::Error(const status_t status, const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printError(&status, format, args);
    va_end(args);
}

template<typename ... X> void DebugLogger::printMessage(const status_t * const status, const char * const format, X... args) const
{
    printHeader(defaultStream, levelMessage);
    print(defaultStream, status, format, args...);
}

template<typename ... X> void DebugLogger::printError(const status_t * const status, const char * const format, X... args) const
{
    printHeader(stderr, levelError);
    print(stderr, status, format, args...);
}

inline void DebugLogger::printHeader(FILE * const streamIn, const char * const level) const
{
    fprintf(streamIn, "%s%s", component, level);
}

template<typename ... X> void DebugLogger::print(FILE * const streamIn, const status_t * const status,
    const char * const format, X... args) const
{
    if (nullptr != status)
    {
        fprintf(streamIn, "Status: %s [%d]%s", StatusToString(*status), *status,
            getStatusDescription(*status));
    }
    if (nullptr != format)
    {
        vfprintf(streamIn, format, args...);
    }
    else
    {
        fprintf(streamIn, "\n");
    }
}

const char * const DebugLogger::getStatusDescription(const intel_gna_status_t status) const
{
    const auto statusSafe = min(status, NUMGNASTATUS);
    return StatusStrings[2 * statusSafe + 1];
}


void VerboseLogger::HorizontalSpacer() const
{
    fprintf(defaultStream, " - ----------------------------------------------------------------\n");
}

const char * const VerboseLogger::getStatusDescription(const intel_gna_status_t status) const
{
    UNREFERENCED_PARAMETER(status);
    return "";
}

#if HW_VERBOSE == 1
std::unique_ptr<Logger> GNA::Log = std::make_unique<VerboseLogger>();
#elif defined(DUMP_ENABLED) || DEBUG >= 1
std::unique_ptr<Logger> GNA::Log = std::make_unique<DebugLogger>();
#else // RELEASE
std::unique_ptr<Logger> GNA::Log = std::make_unique<Logger>();
#endif
