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

#pragma once

#include "HardwareModel.h"
#include "gna-api-verbose.h"

#include <vector>

namespace GNA
{

class SoftwareModel;
class Memory;
class AccelerationDetector;
class Layer;
class RequestConfiguration;
struct RequestProfiler;

class HardwareModelVerbose : public HardwareModel
{
public:
    HardwareModelVerbose(const gna_model_id modId, const std::vector<std::unique_ptr<Layer>>& layers, const Memory& wholeMemory,
        const AccelerationDetector& detector);
    ~HardwareModelVerbose() = default;
    HardwareModelVerbose(const HardwareModelVerbose &) = delete;
    HardwareModelVerbose& operator=(const HardwareModelVerbose&) = delete;

    void SetPrescoreScenario(uint32_t nActions, dbg_action *actions);
    void SetAfterscoreScenario(uint32_t nActions, dbg_action *actions);

    virtual status_t Score(
        uint32_t layerIndex,
        uint32_t layerCount,
        const RequestConfiguration& requestConfiguration,
        RequestProfiler *profiler,
        KernelBuffers *buffers) override final;

private:
    void executeDebugAction(dbg_action action);

    const char * getFileName(dbg_action_type actionType);

    UINT32 readReg(UINT32 regOffset);

    void writeReg(UINT32 regOffset, UINT32 regVal);

    void readPageDir(FILE *f);

    void readRegister(FILE *file, UINT32 registerOffset);

    void writeRegister(dbg_action regAction);

    void dumpMmio(FILE *f);

    void dumpMemory(FILE *file);

    void dumpPageDir(hw_pgdir_out_t &pagedir, FILE *f);

    void dumpPage(uint8_t *ph_addr, uint8_t* v_addr, size_t size, FILE *f);

    void dumpXnnDescriptor(uint16_t layerNumber, FILE *f);

    void setXnnDescriptor(dbg_action action);

    void setGmmDescriptor(dbg_action action);

    void setDescriptor(uint8_t *xnnParam, uint64_t xnnValue, gna_set_size valueSize);

    void zeroMemory(void *memory, size_t memorySize);

    std::vector<dbg_action> prescoreActionVector;

    std::vector<dbg_action> afterscoreActionVector;

    uint32_t pgdirFileNo = 0;
    uint32_t readregFileNo = 0;
    uint32_t mmioFileNo = 0;
    uint32_t xnndescFileNo = 0;
    uint32_t memoryFileNo = 0;
};
}
