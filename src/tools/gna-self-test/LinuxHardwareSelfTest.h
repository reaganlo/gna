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
#pragma once
#include "HardwareSelfTest.h"
#include <vector>
#include "PciDeviceInfo.h"

#define GNA_ST_LSMOD "lsmod | grep ^gna"
#define GNA_ST_MODPROBE "modprobe -v --dry-run gna"

class LinuxGnaSelfTestHardwareStatus : public GnaSelfTestHardwareStatus
{
public:
    LinuxGnaSelfTestHardwareStatus()
    {
        determineUserIdentity();
    }
private:
    void initHardwareInfo() override;
    void initDriverInfo() override;
    int checkHWId();
    int checkDriver();
    std::vector<PciDeviceInfo> getDevicesList();
    // search for a GNA node in /dev/gnaXX - XX in (0,range-1)
    // returns path to the node
    // returns empty string on failure
    const int DEFAULT_GNA_DEV_NODE_RANGE = 16;
    std::string devfsGnaNode(uint8_t range);
    std::string readCmdOutput(const char* command) const;
    void determineUserIdentity() const;
    // end of the search range

};
