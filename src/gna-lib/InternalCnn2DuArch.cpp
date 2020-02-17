/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "GNA_ArchCPkg.h"

#include "gna2-device-api.h"
#include "Expect.h"

// Assuming cpackage required if any device is HW
bool IsCpackageRequired()
{
    Gna2DeviceVersion deviceVersion;
    uint32_t devCount;
    auto status = Gna2DeviceGetCount(&devCount);
    GNA::Expect::Success(status);
    while(devCount-- > 0)
    {
        status = Gna2DeviceGetVersion(devCount, &deviceVersion);
        GNA::Expect::Success(status);
        if (deviceVersion != Gna2DeviceVersionSoftwareEmulation)
        {
            return true;
        }
    }
    return false;
}

GNA3_LyrDesc_t* GNA3_NewLD()
{
    static GNA3_LyrDesc_t t{};
    return &t;
}
void GNA3_FreeLD(GNA3_LyrDesc_t* const t)
{
    const GNA3_LyrDesc_t empty{};
    *t = empty;
}

bool GNA3_PopLD(GNA3_LyrDesc_t* const t)
{
    static const auto required = IsCpackageRequired();
    t->AdaptHW.Valid = !required;
    return t->AdaptHW.Valid;
}
