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

#include <fstream>

#include "AcceleratorHwVerbose.h"
#include "Device.h"
#include "Logger.h"

using std::ifstream;
using std::string;

using namespace GNA;

const string AcceleratorHwVerbose::testCmdFiles[testCmdFilesNo]
{
    "setconfig.txt",
    "setregister.txt",
    "setregister0.txt",
    "setregister1.txt",
    "setregister2.txt",
    "setregister3.txt",
    "setdescriptor.txt"
};

UINT32 AcceleratorHwVerbose::ReadReg(UINT32 regOffset)
{
    status_t status = GNA_SUCCESS;
    hw_read_in_t readRegIn;
    readRegIn.mbarIndex = 0;
    readRegIn.regOffset = regOffset;
    hw_read_out_t readRegOut;
    ZeroMemory(&readRegOut, sizeof(readRegOut));

    sender.IoctlSend(GNA_IOCTL_READ_REG,
        &readRegIn, sizeof(readRegIn),
        &readRegOut, sizeof(readRegOut));

    return readRegOut.regValue;
}

void AcceleratorHwVerbose::WriteReg(UINT32 regOffset, UINT32 regVal)
{
    status_t status = GNA_SUCCESS;
    hw_write_in_t writeRegIn;
    writeRegIn.mbarIndex = 0;
    writeRegIn.regOffset = regOffset;
    writeRegIn.regValue = regVal;

    sender.IoctlSend(GNA_IOCTL_WRITE_REG,
        &writeRegIn, sizeof(writeRegIn),
        nullptr, 0);
}

void AcceleratorHwVerbose::dumpMMIO()
{
	Log->HorizontalSpacer();
    Log->Message("MMIO space\n");
	Log->HorizontalSpacer();
	Log->Message("---                   values (dwords  MSB->LSB)               ---  \n");
	for (int i = 0; i <= 0x118; i += 4)
	{
		Log->Message("%04x %08x\n", i, ReadReg(i));
	}
	Log->LineBreak();
}


#if defined(DUMP_ENABLED)

void AcceleratorHwVerbose::GMMDumpPage(uint8_t* ph_addr, uint8_t* v_addr, size_t size)
{
	Log->HorizontalSpacer();
	for (size_t i = 0; i < size; i += 16)
	{
		Log->Message("%016x    ", (uint64_t)(ph_addr + i));
		Log->Message("%02x", (unsigned int)v_addr[i + 0x03]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x01]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x02]);
		Log->Message("%02x ", (unsigned int)v_addr[i + 0x00]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x07]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x06]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x05]);
		Log->Message("%02x ", (unsigned int)v_addr[i + 0x04]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x0b]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x0a]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x09]);
		Log->Message("%02x ", (unsigned int)v_addr[i + 0x08]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x0f]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x0e]);
		Log->Message("%02x", (unsigned int)v_addr[i + 0x0d]);
		Log->Message("%02x\n", (unsigned int)v_addr[i + 0x0c]);
	}
	Log->LineBreak();
}

void AcceleratorHwVerbose::dumpPageDir()
{
	if (!driverDebug)
	{
	    Log->HorizontalSpacer();
        Log->Message("Page directory DUMP unavailable: driver in release mode.\n");
		return;
	}

	Log->HorizontalSpacer();
    Log->Message("Page directory\n");
	Log->HorizontalSpacer();
	Log->Message("---                  values (dwords MSB->LSB)                 ---  \n");
	for (uint32_t i = 0; i < l1_page_addr.size(); i++)
	{
		Log->Message("entry  %02x %016x\n", i, (l1_page_addr[i] / PAGE_SIZE));
	}
	Log->HorizontalSpacer();
	Log->Message("Page entries\n");
	Log->HorizontalSpacer();
	Log->Message("---           memory dump as dwords(MSB->LSB)        ---  \n");
	for (uint32_t i = 0; i< l1_page_addr.size(); i++)
	{
		GMMDumpPage((uint8_t*)(l1_page_addr[i]), pagedirBuffer + i*PAGE_SIZE, PAGE_SIZE);
	}
	Log->LineBreak();
}

void AcceleratorHwVerbose::dumpPageData()
{
	if (!driverDebug)
	{
	    Log->HorizontalSpacer();
		Log->Message("GMM Data DUMP unavailable: driver in release mode.\n");
		return;
	}

	Log->HorizontalSpacer();
	Log->Message("GMM Data\n");
	Log->HorizontalSpacer();
	Log->Message("---           memory dump as dwords (MSB->LSB)        ---  \n");
	uint32_t i = 0;
	unsigned int* pPhAddr32 = (unsigned int*)pagedirBuffer;
	for (; i < Device::bufferSize / PAGE_SIZE; i++)
	{
#ifdef PHDUMP
		UINT64 phAddr64 = *pPhAddr32++;
		phAddr64 = phAddr64 << 12;

        GMMDumpPage((uint8_t*)phAddr64, ((uint8_t*)Device::buffer) + i*PAGE_SIZE, PAGE_SIZE);
#else
        GMMDumpPage((uint8_t*)(i*PAGE_SIZE), Device::buffer + i*PAGE_SIZE, PAGE_SIZE);
#endif
	}
    if (Device::bufferSize % PAGE_SIZE > 0)
	{
#ifdef PHDUMP
		UINT64 phAddr64 = *pPhAddr32++;
		phAddr64 = phAddr64 << 12;

        GMMDumpPage((uint8_t*)phAddr64, ((uint8_t*)buffer) + i*PAGE_SIZE, Device::bufferSize % PAGE_SIZE);
#else
        GMMDumpPage((uint8_t*)(i*PAGE_SIZE), Device::buffer + i*PAGE_SIZE, Device::bufferSize % PAGE_SIZE);
#endif
	}

}

#endif //DUMP_ENABLED
// TODO: generalize and cleanup modification functions
bool AcceleratorHwVerbose::SetRegister(string path)
{
	ifstream infile;            // set register file
	string   command;           // register command line
	char     operation = '\0';  // operation to perform
	uint32_t   address = 0;     // register address
	uint32_t   value = 0;     // new register value
	uint32_t   old_value = 0;     // previous register value

	infile.open(path);
	if (!infile.is_open()) return false;

	Log->Message("%s\n", path.c_str());
	while (!infile.eof())
	{
		// parse and verify command from line
		getline(infile, command);
		if (3 != sscanf(command.c_str(), "%c,%x,%x", &operation, &address, &value))
		{
			continue; // line does not contain valid command
		}
		if ('S' != operation && 'A' != operation && 'O' != operation)
		{
			Log->Error("Invalid operation in command string.\n");
			continue;
		}
		// perform register operation
		old_value = ReadReg(address);
		if ('A' == operation)
		{
			value = old_value & value;
		}
		else if ('O' == operation)
		{
			value = old_value | value;
		}
		WriteReg(address, value);
		Log->Message("REG: 0x%x changed[%c] from: 0x%x to: 0x%x\n",
			address, operation, old_value, value);
	}
	infile.close();
	return true;
}

// TODO: generalize and cleanup modification functions
bool AcceleratorHwVerbose::SetDescriptor(string path, XNN_LYR* buff, hw_calc_in_t* inData)
{
	ifstream infile;            // set register file
	string   command;           // register command line
	char     operation = '\0';  // operation to perform
	uint32_t   address = 0;     // register address
	uint32_t   value = 0;     // new register value
	uint32_t   old_value = 0;     // previous register value

    if (nullptr == inData) return false;

	infile.open(path);
	if (!infile.is_open()) return false;

    if (nullptr == buff)   return false;

	Log->Message("%s\n", path.c_str());
	while (!infile.eof())
	{
		// parse and verify command from line
		getline(infile, command);
		if (3 != sscanf(command.c_str(), "%c,%x,%x", &operation, &address, &value))
		{
			continue; // line does not contain valid command
		}

		if ('S' != operation && 'A' != operation && 'O' != operation
            || XNN_LYR_DSC_SIZE-1 < address )
		{
			Log->Error("Invalid operation in command string.\n");
			continue;
		}

		// perform operation
		old_value = buff->_char[address];
		if ('A' == operation)
		{
			value = old_value & value;
		}
		else if ('O' == operation)
		{
			value = old_value | value;
		}

		dumpDescriptor(buff);

		buff->_char[address] = value;

		Log->Message("DESCRIPTOR: 0x%x changed[%c] from: 0x%x to: 0x%x.\n",
			address, operation, old_value, value);

		dumpDescriptor(buff);
	}

	infile.close();
	return true;
}

void AcceleratorHwVerbose::HwVerifierMemDump(const char* /*fname*/)
{
// TODO: model buffer as input
//#if DEBUG==1
//    uint8_t* mem = (uint8_t*) Device::buffer;
//    size_t i = 0;
//    FILE *dump = nullptr;
//
//    dump = fopen(fname, "wb");
//    if (nullptr == dump)
//    {
//        Log->Error("Failed to open file.\n");
//        return;
//    }
//
//    Log->Message("Pinned memory dump after scoring.\n");
//    i = fwrite(mem, 1, Device::bufferSize, dump);
//
//    if (i != Device::bufferSize)
//    {
//        Log->Error("Memory dump save failed to write all data (%u/%u).\n", i, Device::bufferSize);
//    }
//
//    fclose(dump);
//    dump = nullptr;
//    Log->Message("Dump complete.\n");
//#endif
}

void AcceleratorHwVerbose::HwVerifier(Request* /*r*/)
{
    //SoftwareModel* model = r->model;
    //HwVerifier(model, ((Hw*)r->handle)->inData->status);
}

void AcceleratorHwVerbose::HwVerifier(SoftwareModel */*model*/, status_t scoring_status) {
    // retrieve output
    Log->Message(scoring_status, "Returned by Ioctl Wait.\n");

    HwVerifierMemDump("dump-after.bin");

#ifdef DUMP_ENABLED
    Log->Message("GMM state after scoring.\n");
    Log->HorizontalSpacer();
    dumpPageDir();
    dumpPageData();
#endif // DUMP_ENABLED

    if (hasTestCmdFile())
    {
        Log->Message("DUMP A, state after regular scoring.\n");
        dumpMMIO();
    }
    if (SetRegister("setregister.txt"))
    {
        // reset all scores
       /* if (model && model->Layers != nullptr && model->layers[model->layerCount - 1].ScratchPad != nullptr)
        {
            Log->Message("Clean outputs in DNN.\n");
            memset(model->layers[model->layerCount - 1].pOutputs, 0, model->layers[model->layerCount - 1].RowCount * 2);
            memset(model->layers[model->layerCount - 1].ScratchPad, 0, model->layers[model->layerCount - 1].RowCount * 4);
        }*/

        Log->Message("DUMP B, before scoring.\n");
        dumpMMIO();
        uint32_t ctr = ReadReg(0x84);
        WriteReg(0x84, ctr | 1);
        Log->Message("scoring started, waiting 2ms...\n");
        Sleep(2);

        if (SetRegister("setregister2.txt"))
        {
            Log->Message("waiting 200ms...\n");
            Sleep(200);
            Log->Message("DUMP C, after 200ms.\n");
            dumpMMIO();
            SetRegister("setregister3.txt");
        }

        Log->Message("checking completion status.\n");
        for (uint8_t i = 0; i < 30; ++i)
        {
            if (0x1 & ReadReg(0x80))
            {
                Log->LineBreak();
                Log->Message("Operation complete.\n");
                break;
            }
            Log->Message("#");
            Sleep(100);
        }
        Log->LineBreak();
        Log->Message("DUMP D, scoring completed.\n");
        dumpMMIO();

        Log->Message("Done, Reset state.\n");
        WriteReg(0xa0, 0);
        WriteReg(0xa4, 0);
        WriteReg(0x80, 0x20000);
        WriteReg(0x84, 4);
    }
}

#define DUMP_CFG(field) Log->Message("%02p %08x\n", &(field), (uint32_t)field)
#define DUMP_CFG_ADDR(pointer) Log->Message("%02p %08x\n", &(pointer), pointer)

void AcceleratorHwVerbose::dumpDescriptor(XNN_LYR* buff)
{
	Log->HorizontalSpacer();
	Log->Message("Descriptor space\n");
	Log->HorizontalSpacer();
	Log->Message("---                   values (dwords  MSB->LSB)               ---\n");
    DUMP_CFG(buff->op);
    DUMP_CFG(buff->flags._char);
    DUMP_CFG(buff->n_in_elems);
    DUMP_CFG(buff->n_out_elems);
    DUMP_CFG(buff->cnn_n_out_p_flt);
    DUMP_CFG(buff->n_groups);
    DUMP_CFG(buff->cnn_n_flt_last);
    DUMP_CFG(buff->n_iters);
    DUMP_CFG(buff->cnn_pool_stride);
    DUMP_CFG(buff->n_elems_last);
    DUMP_CFG(buff->cnn_n_flt_stride);
    DUMP_CFG(buff->rnn_n_fb_iters);
    DUMP_CFG(buff->cnn_pool_size);
    DUMP_CFG(buff->rnn_n_elems_first);
    DUMP_CFG(buff->cnn_n_flts);
    DUMP_CFG(buff->rnn_n_elems_last);
    DUMP_CFG(buff->cnn_n_flt_iters);
    DUMP_CFG(buff->pwl_n_segs);
    DUMP_CFG(buff->act_list_n_elems);
    DUMP_CFG(buff->cpy_n_elems);
    DUMP_CFG(buff->cnn_flt_size);
    DUMP_CFG(buff->cnn_n_flts_iter);
    DUMP_CFG(buff->cnn_n_flt_outs);
    DUMP_CFG(buff->cnn_flt_bf_sz_iter);
    DUMP_CFG(buff->cnn_flt_bf_sz_last);
    DUMP_CFG_ADDR(buff->in_buffer);
    DUMP_CFG_ADDR(buff->out_act_fn_buffer);
    DUMP_CFG_ADDR(buff->out_sum_buffer);
    DUMP_CFG_ADDR(buff->rnn_out_fb_buffer);
    DUMP_CFG_ADDR(buff->aff_weight_buffer);
    DUMP_CFG_ADDR(buff->cnn_flt_buffer);
    DUMP_CFG_ADDR(buff->aff_const_buffer);
    DUMP_CFG_ADDR(buff->act_list_buffer);
    DUMP_CFG_ADDR(buff->pwl_seg_def_buffer);
    DUMP_CFG_ADDR(buff->in_buffer);
    DUMP_CFG_ADDR(buff->in_buffer);
}

void AcceleratorHwVerbose::dumpCfg(uint32_t* config)
{
	Log->HorizontalSpacer();
	Log->Message("Config space\n");
	Log->HorizontalSpacer();
	Log->Message("---                   values (dwords  MSB->LSB)               ---  \n");

	for (int i = 0; i < 32; i++)
	{
		Log->Message("%04x %08x\n", i * 4, config[i]);
	}
}
// TODO: generalize and cleanup modification functions
bool AcceleratorHwVerbose::SetConfig(string path, hw_calc_in_t* inData)
{
	ifstream infile;            // set register file
	string   command;           // register command line
	char     operation = '\0';  // operation to perform
	uint32_t   address = 0;     // register address
	uint32_t   value = 0;     // new register value
	uint32_t   old_value = 0;     // previous register value

    if (nullptr == inData) return false;

	HwVerifierMemDump("dump-before.bin");

	infile.open(path);
	if (!infile.is_open()) return false;

	Log->Message("%s\n", path.c_str());
	//while (!infile.eof())
	//{
	//	// parse and verify command from line
	//	getline(infile, command);
	//	if (3 != sscanf(command.c_str(), "%c,%x,%x", &operation, &address, &value))
	//	{
	//		continue; // line does not contain valid command
	//	}
	//	if ('S' != operation && 'A' != operation && 'O' != operation
 //           || CFG_SIZE - 1 < address)
	//	{
	//		Log->Error("Invalid operation in command string.\n");
	//		continue;
	//	}
 //       address /= sizeof(uint32_t);
	//	// perform register operation
	//	old_value = ((uint32_t*)inData->config)[address]; //ReadReg(address);
	//	if ('A' == operation)
	//	{
	//		value = old_value & value;
	//	}
	//	else if ('O' == operation)
	//	{
	//		value = old_value | value;
	//	}

	//	dumpCfg((uint32_t*)inData->config);
	//	((uint32_t*)inData->config)[address] = value;
	//	Log->Message("CFG: 0x%x changed[%c] from: 0x%x to: 0x%x.\n",
	//		address, operation, old_value, value);

	//	dumpCfg((uint32_t*)inData->config);
	//}

	infile.close();
	return true;
}

bool AcceleratorHwVerbose::hasTestCmdFile()
{
    ifstream testCmdFile;
    uint32_t i = 0;

    for (; i < testCmdFilesNo; i++)
    {
        testCmdFile.open(testCmdFiles[i]);
        if (testCmdFile.is_open())
        {
            testCmdFile.close();
            return true;
        }
    }
    return false;
}