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
#include "PciDeviceInfo.h"
#include <sstream>
#include "SelfTest.h"

std::ostream & _PciDeviceInfo::println(std::ostream & out) const
{
    out << std::hex << domain << ' ' << (int)bus << ' ' << (int)dev << ' ' << (int)func << ' '
        << vendorId << ' ' << deviceId << ' ' << devClass << ' ' << irq << ' ' << (int)irqPin << ' ' << name << std::endl;
    return out;
}

std::string _PciDeviceInfo::toString() const
{
    std::ostringstream oss;
    println(oss);
    return oss.str();
}

// Sample input: "00:03.0 \"0100\" \"1af4\" \"1001\" \"virtio-pci\"
_PciDeviceInfo _PciDeviceInfo::fromLspciString(const std::string & s)
{
    int s_bus, s_dev, s_func, s_vendorId, s_devId, s_devClass;
    int ret = sscanf(s.c_str(), "%x:%x.%d \" %x\" \" %x\" \" %x", &s_bus, &s_dev, &s_func, &s_devClass, &s_vendorId, &s_devId);

    if (ret != 6)
    {
        LOG("ERROR in fromLspciString: parsing error: <%s>\n", s.c_str());
        throw std::logic_error("ERROR in fromLspciString(): parsing error");
    }
    struct _PciDeviceInfo t;
    t.bus = s_bus;
    t.dev = s_dev;
    t.func = s_func;
    t.vendorId = s_vendorId;
    t.deviceId = s_devId;
    t.devClass = s_devClass;
    t.name = s;
    t.domain = 0;
    t.irq = 0;
    t.irqPin = 0;
    return t;
}
