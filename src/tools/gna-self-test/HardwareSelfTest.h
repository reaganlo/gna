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
#include "SelfTest.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <string>

class GnaSelfTestHardwareStatus
{
public:
    GnaSelfTestHardwareStatus(const GnaSelfTest& gst) :gnaSelfTest{ gst }
    {
    }
    virtual ~GnaSelfTestHardwareStatus() = default;
    void Initialize()
    {
        logger.Verbose("Detecting device...\n");
        initHardwareInfo();
        if (hardwareAvailable)
        {
            initDriverInfo();
        }
    }
    bool IsOK() const {
        return (hardwareAvailable && driverAvailable);
    }
    void Print() const
    {
        logger.Verbose("hardwareAvailable: %s\n", hardwareAvailable ? "true" : "false");
        logger.Verbose("driverAvailable: %s\n", driverAvailable ? "true" : "false");
    }
protected:
    bool hardwareAvailable;
    bool driverAvailable;
    void Handle(const GnaSelfTestIssue& issue) const
    {
        gnaSelfTest.Handle(issue);
    }
    const GnaSelfTest& gnaSelfTest;
private:
    virtual void initHardwareInfo() = 0;
    virtual void initDriverInfo() = 0;
};

const std::map<std::pair<uint16_t, uint16_t>, std::string>& GetKnownGNADevsIds();

template<class T>
std::string ToHexString(const T value)
{
    std::stringstream out;
    out << std::setfill('0') << std::setw(sizeof(T) * 2) << std::uppercase << std::hex << value;
    return out.str();
}
