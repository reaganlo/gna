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

#include <string>

#include "AcceleratorHw.h"

namespace GNA
{

#ifndef HW_VERBOSE
#   error   HW_VERBOSE must be defined in 0 or 1 in project configuration or makefile
#endif // HW_VERBOSE

class AcceleratorHwVerbose: public AcceleratorHw
{
public:	
    virtual ~AcceleratorHwVerbose() = default;
    AcceleratorHwVerbose() = delete;
    AcceleratorHwVerbose(const AcceleratorHwVerbose &) = delete;
    AcceleratorHwVerbose& operator=(const AcceleratorHwVerbose&) = delete;

	uint32_t ReadReg(uint32_t regOffset);
	void WriteReg(uint32_t regOffset, uint32_t regVal);

protected:

	/**
     * virtual hw verification methods implemented in HW VERBOSE version only
     * NOTE: DO NOT USE WITH MULTITHREADED APPLICATIONS
     */
    /* @See AcceleratorHw.h */
    bool SetConfig(std::string path, hw_calc_in_t* inData) override;

    /* @See AcceleratorHw.h */
	bool SetDescriptor(std::string path, XNN_LYR* buff, hw_calc_in_t* inData) override;
    
    /* @See AcceleratorHw.h */
    bool SetRegister(std::string path) override;

    /* @See AcceleratorHw.h */
    void HwVerifier(Request* r) override;

    /* @See AcceleratorHw.h */
    void HwVerifier(SoftwareModel *model, status_t scoring_status);

private:

    /**
     * number of supported test command files
     */
    static const uint32_t testCmdFilesNo = 7;

    /**
     * supported test command file names
     */
    static const std::string testCmdFiles[testCmdFilesNo];

    /**
      * Verifies if test command files are available
      *
      * @return true if any test cmd file is available
      */
    bool    hasTestCmdFile();

    void dumpMMIO();

    void dumpCfg(uint32_t* config);

	void dumpDescriptor(XNN_LYR* buff);

	void HwVerifierMemDump(const char* fname);

#if defined(DUMP_ENABLED)

	void dumpPageDir();

	void dumpPageData();

	vector<UINT64> l1_page_addr;
	unsigned char  pagedirBuffer[PAGE_SIZE*GMM_FV_MEM_ALIGN];

	void AcceleratorHw::GMMDumpPage(
		uint8_t* ph_addr,
		uint8_t* v_addr,
		size_t size);

#endif // DUMP_ENABLED

};

}
