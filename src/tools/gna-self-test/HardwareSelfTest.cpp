//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2020 Intel Corporation
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

#include "HardwareSelfTest.h"

#include <cstdint>
#include <map>
#include <string>

const std::map<std::pair<uint16_t, uint16_t>, std::string>& GetKnownGNADevsIds()
{
    static const std::map<std::pair<uint16_t, uint16_t>, std::string> knownGNADevsIds = {
    {{0x8086,0x5A11}, "GNA 0.9 CNL"},
    {{0x8086,0x3190}, "GNA 1.0 GLK"},
    {{0x8086,0x4511}, "GNA 1.0 EHL"},
    {{0x8086,0x8A11}, "GNA 1.0 ICL"},
    {{0x8086,0x9A11}, "GNA 2.0 TGL"},
    {{0x8086,0x4E11}, "GNA 2.0 JSL"},
    //  uncomment for testing purposes then gns-self-test will discover these devices as GNAs
    //  {{0x10ec,0x8168}, "r8169 pseudo GNA device - for testing"}, // Realtek's 8169 Ethernet Controller
    //  {{0x8086,0x2668}, "Android emulator's pseudo GNA device - for testing"}, // Some random Android emulator's device
    };
    return knownGNADevsIds;
}
