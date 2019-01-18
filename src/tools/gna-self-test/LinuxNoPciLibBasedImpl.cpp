//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material contains trade secrets and proprietary
// and confidential information of Intel or its suppliers and licensors.
// The Material is protected by worldwide copyright and trade secret laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed in any way
// without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//*****************************************************************************
#include "LinuxHardwareSelfTest.h"
#include <sstream>

// List PCI devices in machine readable format (-m), includieng kernel driver info (-k) and numeric form (-n)
#define GNA_ST_LSPCI "lspci -n -k -m"

std::vector<PciDeviceInfo> LinuxGnaSelfTestHardwareStatus::getDevicesList()
{
    logger.Verbose("INFO in getDevicesList LSPCI method\n");
    std::vector<PciDeviceInfo> devList;
    std::istringstream in{ readCmdOutput(GNA_ST_LSPCI)};
    std::string s;
    while(std::getline(in,s))
    {
        if(s.size()<10) continue;  //too short to be 'a proper line' from lspci
        PciDeviceInfo dev = PciDeviceInfo::fromLspciString(s);
        devList.push_back(dev);
    }

    return devList;
}
