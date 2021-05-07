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
#include <string>
#include <cstdint>
typedef struct _PciDeviceInfo {
    std::string name;
    uint16_t  domain;
    uint8_t bus;
    uint8_t dev;
    uint8_t func;
    uint16_t vendorId;
    uint16_t deviceId;
    uint16_t devClass;
    int irq;
    uint8_t irqPin;
    std::ostream& println(std::ostream& out) const;
    std::string toString() const;
    // Sample input: "00:03.0 \"0100\" \"1af4\" \"1001\" \"virtio-pci\"
    static struct _PciDeviceInfo fromLspciString(const std::string& s);
} PciDeviceInfo;