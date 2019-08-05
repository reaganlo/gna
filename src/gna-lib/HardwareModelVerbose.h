/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#include "HardwareModelScorable.h"
#include "gna-api-verbose.h"

#if defined(_WIN32)
#if HW_VERBOSE == 1
#include "GnaDrvApiWinDebug.h"
#else
#include "GnaDrvApiWin.h"
#endif
#else
#error Verbose version of library available only on Windows OS
#endif

#include <vector>

namespace GNA
{

class AccelerationDetector;
class CompiledModel;
class Layer;
class Memory;
class RequestConfiguration;
class RequestProfiler;
class SoftwareModel;
class WindowsDriverInterface;

class HardwareModelVerbose : public HardwareModelScorable
{
public:
    HardwareModelVerbose(CompiledModel const & softwareModel,
        DriverInterface &ddi, const HardwareCapabilities& hwCaps);
    virtual ~HardwareModelVerbose() = default;
    HardwareModelVerbose(const HardwareModelVerbose &) = delete;
    HardwareModelVerbose& operator=(const HardwareModelVerbose&) = delete;

    void SetPrescoreScenario(uint32_t nActions, dbg_action *actions);
    void SetAfterscoreScenario(uint32_t nActions, dbg_action *actions);

    virtual uint32_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers) override final;

private:
    static void zeroMemory(void *memoryIn, size_t memorySize);

    void executeDebugAction(dbg_action& action);

    FILE * const getActionFile(dbg_action& action);

    UINT32 readReg(UINT32 regOffset);

    void writeReg(UINT32 regOffset, UINT32 regVal);

    void readRegister(FILE *file, UINT32 registerOffset);

    void writeRegister(dbg_action regAction);

    void dumpMmio(FILE *f);

    void dumpMemory(FILE *file);

    void dumpXnnDescriptor(uint32_t layerNumber, FILE *f);

    void dumpGmmDescriptor(uint32_t layerNumber, FILE *f);

    void setXnnDescriptor(dbg_action action);

    void setGmmDescriptor(dbg_action action);

    void setDescriptor(uint8_t *xnnParam, uint64_t xnnValue, gna_set_size valueSize);

    std::vector<dbg_action> prescoreActionVector;

    std::vector<dbg_action> afterscoreActionVector;

    size_t memorySize;

    static std::map<dbg_action_type const, char const * const> const actionFileNames;
    std::map<dbg_action_type const, uint32_t> actionFileCounters;

    WindowsDriverInterface &windowsDriverInterface;
};
}
