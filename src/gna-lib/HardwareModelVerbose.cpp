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

#include "CompiledModel.h"
#include "HardwareLayer.h"
#include "HardwareModelVerbose.h"
#include "Layer.h"
#include "Memory.h"
#include "MemoryContainer.h"
#include "WindowsDriverInterface.h"

#if defined(_WIN32)
#if HW_VERBOSE == 1
#include "GnaDrvApiWinDebug.h"
#endif
#else
#error Verbose version of library available only on Windows OS
#endif

#include <fstream>
#include <memory>
#include <string>

using namespace GNA;

std::map<dbg_action_type const, char const * const> const HardwareModelVerbose::actionFileNames =
{
    {GnaDumpMmio, "dumpmmio_"},
    {GnaReadRegister, "readreg_"},
    {GnaDumpMemory, "memory_"},
    {GnaDumpXnnDescriptor, "xnndesc_"},
    {GnaDumpGmmDescriptor, "gmmdesc_"}
};

HardwareModelVerbose::HardwareModelVerbose(CompiledModel const & softwareModel,
        DriverInterface &ddi, const HardwareCapabilities& hwCaps) :
    HardwareModelScorable(softwareModel, ddi, hwCaps),
    actionFileCounters{
        {GnaDumpMmio, 0},
        {GnaReadRegister, 0},
        {GnaDumpMemory, 0},
        {GnaDumpXnnDescriptor, 0},
        {GnaDumpGmmDescriptor, 0}},
    windowsDriverInterface{dynamic_cast<WindowsDriverInterface&>(ddi)}
{
}

uint32_t HardwareModelVerbose::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler &profiler,
    KernelBuffers *buffers)
{
    for (auto& action : prescoreActionVector)
    {
        executeDebugAction(action);
    }

    auto status = HardwareModelScorable::Score(layerIndex, layerCount,
                        requestConfiguration, profiler, buffers);

    for (auto& action : afterscoreActionVector)
    {
        executeDebugAction(action);
    }

    return status;
}

void HardwareModelVerbose::SetPrescoreScenario(uint32_t nActions, dbg_action *actions)
{
    prescoreActionVector.clear();
    prescoreActionVector.insert(prescoreActionVector.begin(), actions, actions + nActions);
}

void HardwareModelVerbose::SetAfterscoreScenario(uint32_t nActions, dbg_action *actions)
{
    afterscoreActionVector.clear();
    afterscoreActionVector.insert(afterscoreActionVector.begin(), actions, actions + nActions);
}

UINT32 HardwareModelVerbose::readReg(UINT32 regOffset)
{
    GNA_READREG_IN readRegIn;
    readRegIn.mbarIndex = 0;
    readRegIn.regOffset = regOffset;
    GNA_READREG_OUT readRegOut;
    ZeroMemory(&readRegOut, sizeof(readRegOut));

    windowsDriverInterface.IoctlSend(GNA_COMMAND_READ_REG,
        &readRegIn, sizeof(readRegIn),
        &readRegOut, sizeof(readRegOut));

    return readRegOut.regValue;
}

void HardwareModelVerbose::writeReg(UINT32 regOffset, UINT32 regVal)
{
    GNA_WRITEREG_IN writeRegIn;
    writeRegIn.mbarIndex = 0;
    writeRegIn.regOffset = regOffset;
    writeRegIn.regValue = regVal;

    windowsDriverInterface.IoctlSend(GNA_COMMAND_WRITE_REG,
        &writeRegIn, sizeof(writeRegIn),
        NULL, 0);
}

void HardwareModelVerbose::readRegister(FILE *file, UINT32 registerOffset)
{
    fprintf(file, "%08x\n", readReg(registerOffset));
}

void HardwareModelVerbose::writeRegister(dbg_action regAction)
{
    uint32_t regValue = 0;
    switch (regAction.reg_params.reg_operation)
    {
    case Equal:
        regValue = regAction.reg_params.reg_value;
        break;
    case And:
        regValue = readReg(regAction.reg_params.gna_register);
        regValue &= regAction.reg_params.reg_value;
        break;
    case Or:
        regValue = readReg(regAction.reg_params.gna_register);
        regValue |= regAction.reg_params.reg_value;
        break;
    }
    writeReg(regAction.reg_params.gna_register, regValue);
}

void HardwareModelVerbose::dumpMemory(FILE *file)
{
    allocations.WriteData(file);
}

void HardwareModelVerbose::zeroMemory(void *memoryIn, size_t memorySizeIn)
{
    memset(memoryIn, 0, memorySizeIn);
}

void HardwareModelVerbose::setXnnDescriptor(dbg_action action)
{
    auto & hwLayer = GetLayer(action.xnn_params.layer_number);
    auto const xnnParam = hwLayer.XnnDescriptor.GetMemAddress() +
                        action.xnn_params.xnn_offset;
    setDescriptor(xnnParam, action.xnn_params.xnn_value, action.xnn_params.xnn_value_size);
}

void HardwareModelVerbose::setGmmDescriptor(dbg_action action)
{
    auto & hwLayer = GetLayer(action.xnn_params.layer_number);
    if (hwLayer.SoftwareLayer.Operation != INTEL_GMM)
    {
        throw GnaException{ Gna2StatusXnnErrorLyrOperation };
    }
    auto const gmmDescriptor = hwLayer.XnnDescriptor.GmmDescriptor;
    auto const xnnParam = gmmDescriptor.Get<uint8_t>() + action.xnn_params.xnn_offset;
    setDescriptor(xnnParam, action.xnn_params.xnn_value, action.xnn_params.xnn_value_size);
}

void HardwareModelVerbose::setDescriptor(uint8_t *xnnParam, uint64_t xnnValue, gna_set_size valueSize)
{
    switch (valueSize)
    {
    case GNA_SET_BYTE:
        *xnnParam = static_cast<uint8_t>(xnnValue);
        break;
    case GNA_SET_WORD:
        *reinterpret_cast<uint16_t*>(xnnParam) = static_cast<uint16_t>(xnnValue);
        break;
    case GNA_SET_DWORD:
        *reinterpret_cast<uint32_t*>(xnnParam) = static_cast<uint32_t>(xnnValue);
        break;
    case GNA_SET_QWORD:
        *reinterpret_cast<uint64_t*>(xnnParam) = xnnValue;
        break;
    case GNA_SET_XNNLYR:
        memset(xnnParam, static_cast<int>(xnnValue), baseDescriptor->GetSize());
        break;
    }
}

void HardwareModelVerbose::dumpMmio(FILE *file)
{
    fprintf(file, "\nMMIO space\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---                   values (dwords  MSB->LSB)               ---  \n");
    for (int i = GNA_STS; i <= GNA_SAIV; i += sizeof(int32_t))
    {
        fprintf(file, "%04x %08x\n", i, readReg(i));
    }
    fprintf(file, "\n");
}

FILE * const GNA::HardwareModelVerbose::getActionFile(dbg_action & action)
{
    // determine if actionFile is necessary
    if (0 == actionFileNames.count(action.action_type))
    {
        return nullptr; // no, return nullptr
    }
    // yes, open actionFile
    FILE * actionFile = nullptr;
    errno_t openError = 0;
    try
    {
        // get actionFile name
        std::string actionFileName;
        if (nullptr != action.filename)
        {
            actionFileName = std::string(action.filename);
        }
        if (actionFileName.empty())
        {
            actionFileName = actionFileNames.at(action.action_type)
                + std::to_string(actionFileCounters[action.action_type]++) + ".txt";
        }
        if (actionFileName.empty())
        {
            throw GnaException(Gna2StatusNullArgumentNotAllowed);
        }

        // open file
        fopen_s(&actionFile, actionFileName.c_str(), "w");
        if (0 != openError || nullptr == actionFile)
        {
            throw GnaException(Gna2StatusNullArgumentNotAllowed);
        }
    }
    catch (...)
    {
        Log->Error("Action File is missing.\n");
        throw;
    }
    return actionFile;
}

void HardwareModelVerbose::executeDebugAction(dbg_action& action)
{
    FILE * file = nullptr;
    try
    {
        file = getActionFile(action);

        switch (action.action_type)
        {
        case GnaDumpMmio:
        dumpMmio(file);
        break;
        case GnaReadRegister:
        readRegister(file, action.reg_params.gna_register);
        break;
        case GnaWriteRegister:
        writeRegister(action);
        break;
        case GnaDumpMemory:
        dumpMemory(file);
        break;
        case GnaZeroMemory:
        zeroMemory(action.output_params.outputs, action.output_params.outputs_size);
        break;
        case GnaDumpXnnDescriptor:
        dumpXnnDescriptor(action.xnn_params.layer_number, file);
        break;
        case GnaDumpGmmDescriptor:
        dumpGmmDescriptor(action.xnn_params.layer_number, file);
        break;
        case GnaSetXnnDescriptor:
        setXnnDescriptor(action);
        break;
        case GnaSetGmmDescriptor:
        setGmmDescriptor(action);
        break;
        case GnaLogMessage:
        Log->Message("%s", action.log_message);
        break;
        case GnaSleep:
        Sleep(action.timeout);
        break;
        }

        if (nullptr != file)
        {
            fclose(file);
        }
    }
    catch (...)
    {
        if (nullptr != file)
        {
            fclose(file);
        }
        throw;
    }
}

#define DUMP_CFG(file, field) fprintf(file, "%02p %08x\n", &(field), field.Get())
#define DUMP_CFG_ADDR(file, pointer) fprintf(file, "%02p %08x\n", &(pointer), pointer.Get())
#define DUMP_GMM_ADDR(file, pointer) fprintf(file, "%02p %08x\n", &(pointer), pointer)

void HardwareModelVerbose::dumpXnnDescriptor(uint32_t layerNumber, FILE *file)
{
    auto & lyrDsc = GetLayer(layerNumber).XnnDescriptor;
    fprintf(file, "\nDescriptor space\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---                   values (dwords  MSB->LSB)               ---\n");
    DUMP_CFG(file, lyrDsc[op]);
    DUMP_CFG(file, lyrDsc[flags]);
    DUMP_CFG(file, lyrDsc[n_in_elems]);
    DUMP_CFG(file, lyrDsc[n_out_elems]);
    DUMP_CFG(file, lyrDsc[cnn_n_out_p_flt]);
    DUMP_CFG(file, lyrDsc[n_groups]);
    DUMP_CFG(file, lyrDsc[cnn_n_flt_last]);
    DUMP_CFG(file, lyrDsc[n_iters]);
    DUMP_CFG(file, lyrDsc[cnn_pool_stride]);
    DUMP_CFG(file, lyrDsc[n_elems_last]);
    DUMP_CFG(file, lyrDsc[cnn_n_flt_stride]);
    DUMP_CFG(file, lyrDsc[rnn_n_fb_iters]);
    DUMP_CFG(file, lyrDsc[cnn_pool_size]);
    DUMP_CFG(file, lyrDsc[rnn_n_elems_first]);
    DUMP_CFG(file, lyrDsc[cnn_n_flts]);
    DUMP_CFG(file, lyrDsc[rnn_n_elems_last]);
    DUMP_CFG(file, lyrDsc[cnn_n_flt_iters]);
    DUMP_CFG(file, lyrDsc[pwl_n_segs]);
    DUMP_CFG(file, lyrDsc[act_list_n_elems]);
    DUMP_CFG(file, lyrDsc[cpy_n_elems]);
    DUMP_CFG(file, lyrDsc[cnn_flt_size]);
    DUMP_CFG(file, lyrDsc[cnn_n_flts_iter]);
    DUMP_CFG(file, lyrDsc[cnn_n_flt_outs]);
    DUMP_CFG(file, lyrDsc[cnn_flt_bf_sz_iter]);
    DUMP_CFG(file, lyrDsc[cnn_flt_bf_sz_last]);
    DUMP_CFG_ADDR(file, lyrDsc[in_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[out_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[out_sum_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[rnn_out_fb_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[weight_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[bias_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[act_list_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[pwl_seg_def_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[in_buffer]);
    DUMP_CFG_ADDR(file, lyrDsc[in_buffer]);
}

void HardwareModelVerbose::dumpGmmDescriptor(uint32_t layerNumber, FILE *file)
{
    fprintf(file, "\nGMM Descriptor space\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---                   values (dwords  MSB->LSB)               ---\n");

    auto const & gmmConfig = GetLayer(layerNumber).GmmDescriptor;
    for (size_t i = 0; i < sizeof(GMM_CONFIG) / sizeof(uint32_t); i++)
    {
        DUMP_GMM_ADDR(file, (gmmConfig.Get()->_value[i]));
    }
}
