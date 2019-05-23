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

#include "common.h"
#include "gna-api-status.h"

#include <memory>
#include <stdio.h>

namespace GNA
{

// Release build Logger serving mainly as interface
struct Logger
{
    Logger() = default;
    virtual ~Logger() = default;

    virtual void LineBreak() const;
    virtual void HorizontalSpacer() const;

    virtual void Message(const status_t status) const;
    virtual void Message(const status_t status, const char * const format, ...) const;
    virtual void Message(const char * const format, ...) const;

    virtual void Warning(const char * const format, ...) const;

    virtual void Error(const char * const format, ...) const;
    virtual void Error(const status_t status, const char * const format, ...) const;
    virtual void Error(const status_t status) const;

    static const char * StatusToString(const intel_gna_status_t status) noexcept;

protected:
    static const char * const StatusStrings[2 * (NUMGNASTATUS + 1)];

    Logger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
        const char * const levelErrorIn);

    FILE * const defaultStream = stdout;
    const char * const component = "";
    const char * const levelMessage = "INFO: ";
    const char * const levelWarning = "WARNING: ";
    const char * const levelError = "ERROR: ";
};

// Logger for debug builds
struct DebugLogger : public Logger
{
    DebugLogger() :
        DebugLogger(stderr, "[IntelGna] ", "INFO: ", "ERROR: ")
    {}
    virtual ~DebugLogger() = default;

    virtual void LineBreak() const override;

    virtual void Message(const status_t status) const override;
    virtual void Message(const status_t status, const char * const format, ...) const override;
    virtual void Message(const char * const format, ...) const override;

    virtual void Warning(const char * const format, ...) const override;


    virtual void Error(const status_t status) const override;
    virtual void Error(const char * const format, ...) const override;
    virtual void Error(const status_t status, const char * const format, ...) const override;

protected:
    template<typename ... X> void printMessage(
        const status_t * const status, const char * const format, X... args) const;
    template<typename ... X> void printWarning(const char * const format, X... args) const;
    template<typename ... X>  void printError(
        const status_t * const status, const char * const format, X... args) const;
    inline void printHeader(FILE * const streamIn, const char * const level) const;
    template<typename ... X>  void print(
        FILE * const streamIn, const status_t * const status, const char * const format, X... args) const;

    DebugLogger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
        const char * const levelErrorIn);

    virtual const char * getStatusDescription(const intel_gna_status_t status) const;
};

// Verbose logger for validation purposes
struct VerboseLogger : public DebugLogger
{
    VerboseLogger() :
        DebugLogger()
    {}
    virtual ~VerboseLogger() = default;

    virtual void HorizontalSpacer() const override;

protected:
    virtual const char * getStatusDescription(const intel_gna_status_t status) const override;
};

extern std::unique_ptr<Logger> Log;

}
