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

#include "HardwareModelVerbose.h"

#include <fstream>
#include <string>

using namespace GNA;


std::map<dbg_action_type const, char const * const> const HardwareModelVerbose::actionFileNames = 
{
    {GnaDumpMmio, "dumpmmio_"},
    {GnaDumpPageDirectory, "dumppgdir_"},
    {GnaReadRegister, "readreg_"},
    {GnaDumpMemory, "memory_"},
    {GnaDumpXnnDescriptor, "xnndesc_"},
    {GnaDumpGmmDescriptor, "gmmdesc_"}
};

HardwareModelVerbose::HardwareModelVerbose(const gna_model_id modId,
    const std::vector<std::unique_ptr<Layer>>& layers, uint16_t gmmCount, const Memory &memoryIn,
    IoctlSender &sender, const AccelerationDetector& detector) :
    HardwareModel::HardwareModel(modId, layers, gmmCount, memoryIn.Id, memoryIn.Get(), memoryIn.GetDescriptorsBase(modId), sender, detector),
    memorySize{ memoryIn.GetSize() },
    actionFileCounters{
        {GnaDumpMmio, 0},
        {GnaDumpPageDirectory, 0},
        {GnaReadRegister, 0},
        {GnaDumpMemory, 0},
        {GnaDumpXnnDescriptor, 0},
        {GnaDumpGmmDescriptor, 0}}
{
}

status_t HardwareModelVerbose::Score(
    uint32_t layerIndex,
    uint32_t layerCount,
    const RequestConfiguration& requestConfiguration,
    RequestProfiler *profiler,
    KernelBuffers *buffers,
    const GnaOperationMode operationMode)
{
    for (auto& action : prescoreActionVector)
    {
        executeDebugAction(action);
    }

    auto status = HardwareModel::Score(layerIndex, layerCount, requestConfiguration, profiler, buffers, operationMode);

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

void HardwareModelVerbose::readPageDir(FILE *file)
{
    hw_mmap_in_t readPageDirIn;
    readPageDirIn.memoryId = memoryId;

    hw_pgdir_out_t readPageDirOut;
    ZeroMemory(&readPageDirOut, sizeof(readPageDirOut));

    ioctlSender.IoctlSend(GNA_IOCTL_READ_PGDIR,
        &readPageDirIn, sizeof(readPageDirIn),
        &readPageDirOut, sizeof(readPageDirOut));

    dumpPageDir(readPageDirOut, file);
}

void HardwareModelVerbose::dumpPageDir(hw_pgdir_out_t &pagedir, FILE *file)
{
    fprintf(file, "\nPage directory\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---                  values (dwords MSB->LSB)                 ---  \n");
    for (uint32_t i = 0; i < pagedir.ptCount; i++)
    {
        fprintf(file, "entry  %02x %016llx\n", i, (pagedir.l1PhysAddr[i] / PAGE_SIZE));
    }
    fprintf(file, "\nPage entries\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---           memory dump as dwords(MSB->LSB)        ---  \n");
    for (uint32_t i = 0; i < pagedir.ptCount; i++)
    {
        dumpPage((uint8_t*)(pagedir.l1PhysAddr[i]), ((uint8_t*)(pagedir.l2PhysAddr)) + i*PAGE_SIZE, PAGE_SIZE, file);
    }
    fprintf(file, "\n");
}

void HardwareModelVerbose::dumpPage(uint8_t *ph_addr, uint8_t* v_addr, size_t size, FILE *file)
{
    fprintf(file, "\n-----------------------------------------------------------------\n");
    for (size_t i = 0; i < size; i += 16)
    {
        fprintf(file, "%016llx    ", (uint64_t)(ph_addr + i));
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x03]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x01]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x02]);
        fprintf(file, "%02x ", (unsigned int)v_addr[i + 0x00]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x07]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x06]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x05]);
        fprintf(file, "%02x ", (unsigned int)v_addr[i + 0x04]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x0b]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x0a]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x09]);
        fprintf(file, "%02x ", (unsigned int)v_addr[i + 0x08]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x0f]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x0e]);
        fprintf(file, "%02x", (unsigned int)v_addr[i + 0x0d]);
        fprintf(file, "%02x\n", (unsigned int)v_addr[i + 0x0c]);
    }
    fprintf(file, "\n");
}

UINT32 HardwareModelVerbose::readReg(UINT32 regOffset)
{
    hw_read_in_t readRegIn;
    readRegIn.mbarIndex = 0;
    readRegIn.regOffset = regOffset;
    hw_read_out_t readRegOut;
    ZeroMemory(&readRegOut, sizeof(readRegOut));

    ioctlSender.IoctlSend(GNA_IOCTL_READ_REG,
        &readRegIn, sizeof(readRegIn),
        &readRegOut, sizeof(readRegOut));

    return readRegOut.regValue;
}

void HardwareModelVerbose::writeReg(UINT32 regOffset, UINT32 regVal)
{
    hw_write_in_t writeRegIn;
    writeRegIn.mbarIndex = 0;
    writeRegIn.regOffset = regOffset;
    writeRegIn.regValue = regVal;

    ioctlSender.IoctlSend(GNA_IOCTL_WRITE_REG,
        &writeRegIn, sizeof(writeRegIn),
        NULL, 0);
}

void HardwareModelVerbose::readRegister(FILE *file, UINT32 registerOffset)
{
    fprintf(file, "%08x\n", readReg(registerOffset));
}

void HardwareModelVerbose::writeRegister(dbg_action regAction)
{
    uint32_t regValue;
    switch (regAction.reg_operation)
    {
    case Equal:
        regValue = regAction.reg_value;
        break;
    case And:
        regValue = readReg(regAction.gna_register);
        regValue &= regAction.reg_value;
        break;
    case Or:
        regValue = readReg(regAction.gna_register);
        regValue |= regAction.reg_value;
        break;
    }
    writeReg(regAction.gna_register, regValue);
}

void HardwareModelVerbose::dumpMemory(FILE *file)
{
    fwrite(memoryBase.Get(), memorySize, 1, file);
}

void HardwareModelVerbose::zeroMemory(void *memoryIn, size_t memorySizeIn)
{
    memset(memoryIn, 0, memorySizeIn);
}

void HardwareModelVerbose::setXnnDescriptor(dbg_action action)
{
    auto xnnLyr = reinterpret_cast<uint8_t*>(hardwareLayers.at(action.layer_number)->XnnDescriptor);
    auto xnnParam = xnnLyr + action.xnn_offset;
    setDescriptor(xnnParam, action.xnn_value, action.xnn_value_size);
}

void HardwareModelVerbose::setGmmDescriptor(dbg_action action)
{
    auto gmmConfig = reinterpret_cast<uint8_t*>(hardwareLayers.at(action.layer_number)->GmmDescriptor);
    auto xnnParam = gmmConfig + action.xnn_offset;
    setDescriptor(xnnParam, action.xnn_value, action.xnn_value_size);
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
        memset(xnnParam, xnnValue, sizeof(XNN_LYR));
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
            throw GnaException(GNA_NULLARGNOTALLOWED);
        }

        // open file
        fopen_s(&actionFile, actionFileName.c_str(), "w");
        if (0 != openError || nullptr == actionFile)
        {
            throw GnaException(GNA_NULLARGNOTALLOWED);
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
        case GnaDumpPageDirectory:
        readPageDir(file);
        break;
        case GnaReadRegister:
        readRegister(file, action.gna_register);
        break;
        case GnaWriteRegister:
        writeRegister(action);
        break;
        case GnaDumpMemory:
        dumpMemory(file);
        break;
        case GnaZeroMemory:
        zeroMemory(action.outputs, action.outputs_size);
        break;
        case GnaDumpXnnDescriptor:
        dumpXnnDescriptor(action.layer_number, file);
        break;
        case GnaDumpGmmDescriptor:
        dumpGmmDescriptor(action.layer_number, file);
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

#define DUMP_CFG(file, field) fprintf(file, "%02p %08x\n", &(field), (uint32_t)field)
#define DUMP_CFG_ADDR(file, pointer) fprintf(file, "%02p %08x\n", &(pointer), pointer)

void HardwareModelVerbose::dumpXnnDescriptor(uint16_t layerNumber, FILE *file)
{
    auto lyrDsc = (AddrXnnLyr(descriptorsAddress) + layerNumber).Get();

    fprintf(file, "\nDescriptor space\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---                   values (dwords  MSB->LSB)               ---\n");
    DUMP_CFG(file, lyrDsc->op);
    DUMP_CFG(file, lyrDsc->flags._char);
    DUMP_CFG(file, lyrDsc->n_in_elems);
    DUMP_CFG(file, lyrDsc->n_out_elems);
    DUMP_CFG(file, lyrDsc->cnn_n_out_p_flt);
    DUMP_CFG(file, lyrDsc->n_groups);
    DUMP_CFG(file, lyrDsc->cnn_n_flt_last);
    DUMP_CFG(file, lyrDsc->n_iters);
    DUMP_CFG(file, lyrDsc->cnn_pool_stride);
    DUMP_CFG(file, lyrDsc->n_elems_last);
    DUMP_CFG(file, lyrDsc->cnn_n_flt_stride);
    DUMP_CFG(file, lyrDsc->rnn_n_fb_iters);
    DUMP_CFG(file, lyrDsc->cnn_pool_size);
    DUMP_CFG(file, lyrDsc->rnn_n_elems_first);
    DUMP_CFG(file, lyrDsc->cnn_n_flts);
    DUMP_CFG(file, lyrDsc->rnn_n_elems_last);
    DUMP_CFG(file, lyrDsc->cnn_n_flt_iters);
    DUMP_CFG(file, lyrDsc->pwl_n_segs);
    DUMP_CFG(file, lyrDsc->act_list_n_elems);
    DUMP_CFG(file, lyrDsc->cpy_n_elems);
    DUMP_CFG(file, lyrDsc->cnn_flt_size);
    DUMP_CFG(file, lyrDsc->cnn_n_flts_iter);
    DUMP_CFG(file, lyrDsc->cnn_n_flt_outs);
    DUMP_CFG(file, lyrDsc->cnn_flt_bf_sz_iter);
    DUMP_CFG(file, lyrDsc->cnn_flt_bf_sz_last);
    DUMP_CFG_ADDR(file, lyrDsc->in_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->out_act_fn_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->out_sum_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->rnn_out_fb_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->aff_weight_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->cnn_flt_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->aff_const_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->act_list_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->pwl_seg_def_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->in_buffer);
    DUMP_CFG_ADDR(file, lyrDsc->in_buffer);
}

void HardwareModelVerbose::dumpGmmDescriptor(uint16_t layerNumber, FILE *file)
{
    fprintf(file, "\nGMM Descriptor space\n");
    fprintf(file, "-----------------------------------------------------------------\n");
    fprintf(file, "---                   values (dwords  MSB->LSB)               ---\n");
    
    auto gmmConfig = hardwareLayers.at(layerNumber)->GmmDescriptor;
    for (size_t i = 0; i < sizeof(GMM_CONFIG) / sizeof(uint32_t); i++)
    {
        DUMP_CFG_ADDR(file, gmmConfig->_value[i]);
    }
}
