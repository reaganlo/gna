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

#ifndef WIN32

#include "LinuxHwModuleInterface.hpp"

#include "Expect.h"
#include "Logger.h"

#include <dlfcn.h>

using namespace GNA;

LinuxHwModuleInterface::LinuxHwModuleInterface(char const * moduleName)
{
    auto fullName = std::string("./");
    fullName.append(moduleName);
    auto debugName = fullName;
    fullName.append(".so");
    hwModule = dlopen(fullName.c_str(), RTLD_NOW);
    if (nullptr != hwModule)
    {
        Log->Warning("HwModule release library not found, trying to load debug library.\n");
        debugName.append("d.so");
        hwModule = dlopen(debugName.c_str(), RTLD_NOW);
    }
    if (nullptr != hwModule)
    {
        CreateLD = reinterpret_cast<CreateLDFunction>(dlsym(hwModule, "GNA3_NewLD"));
        FillLD = reinterpret_cast<FillLDFunction>(dlsym(hwModule, "GNA3_PopLD"));
        FreeLD = reinterpret_cast<FreeLDFunction>(dlsym(hwModule, "GNA3_FreeLD"));
        Validate();
    }
    else
    {
        Log->Warning("HwModule library not found.\n");
    }
}

LinuxHwModuleInterface::~LinuxHwModuleInterface()
{
    if (nullptr != hwModule)
    {
        auto const error = dlclose(hwModule);
        if (error)
        {
            Log->Error("FreeLibrary failed!\n");
        }
    }
}

#endif
