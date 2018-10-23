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
#include "WindowsHardwareSelfTest.h"
#include <map>
static const std::map<std::string, std::string> validHwIds{
    {"5A11", "GNA 0.9 CNL"},
    {"3190", "GNA 1.0 GLK"},
    {"8A11", "GNA 1.0 ICL"},
    {"9A11", "GNA 2.0 TGL"} };

void WindowsGnaSelfTestHardwareStatus::initHardwareInfo()
{
    hardwareAvailable = (checkHWId() == 0);
}

void WindowsGnaSelfTestHardwareStatus::initDriverInfo()
{
    driverAvailable = (checkDriver() == 0);
}

int WindowsGnaSelfTestHardwareStatus::checkHWId()
{
    GUID cls;
    devInfo = SetupDiGetClassDevsEx(&cls, NULL, NULL, (0 ? 0 : DIGCF_ALLCLASSES), NULL, NULL, NULL);

    devListData.cbSize = sizeof(SP_DEVINFO_LIST_DETAIL_DATA);

    if (!SetupDiGetDeviceInfoListDetail(devInfo, &devListData))
    {
        LOG("SetupDiGetDeviceInfoListDetail FAILED with GetLastError() = %llu\n", (unsigned long long)GetLastError());
        LOG("Make sure to run self-test as Administrator\n");
        GnaSelfTestIssue::SETUPDI_ERROR.Handle();
    }

    devInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

    uint32_t nDevices = 0;
    std::string fullHwId;
    while (SetupDiEnumDeviceInfo(devInfo, nDevices++, &devInfoData))
    {
        LPTSTR *hwIds = NULL;
        uint32_t dataType;
        char *buffer = new char[4096];
        SetupDiGetDeviceRegistryProperty(devInfo, &devInfoData, SPDRP_HARDWAREID, (PDWORD)&dataType, (LPBYTE)buffer, 4096, NULL);
        std::string devHwId(buffer);
        delete[] buffer;

        for (auto vHwId : validHwIds) {
            if (devHwId.rfind(vHwId.first) != std::string::npos) {
                fullHwId = vHwId.second;
                break;
            }
        }

        if (!fullHwId.empty()) {
            LOG("The device have been found: (%s)\n", fullHwId.c_str());
            break;
        }
    }

    if (fullHwId.empty()) {
        LOG("FAILED (no hardware device detected)\n");
        return -1;
    }
    return 0;
}

int WindowsGnaSelfTestHardwareStatus::checkDriver()
{
    SP_DRVINFO_DATA driverInfoData;
    SP_DEVINSTALL_PARAMS deviceInstallParams;
    ZeroMemory(&driverInfoData, sizeof(driverInfoData));
    ZeroMemory(&deviceInstallParams, sizeof(deviceInstallParams));
    driverInfoData.cbSize = sizeof(SP_DRVINFO_DATA);
    deviceInstallParams.cbSize = sizeof(SP_DEVINSTALL_PARAMS);

    SetupDiGetDeviceInstallParams(devInfo, &devInfoData, &deviceInstallParams);
    deviceInstallParams.FlagsEx |= (DI_FLAGSEX_INSTALLEDDRIVER | DI_FLAGSEX_ALLOWEXCLUDEDDRVS);
    SetupDiSetDeviceInstallParams(devInfo, &devInfoData, &deviceInstallParams);

    SetupDiBuildDriverInfoList(devInfo, &devInfoData, SPDIT_COMPATDRIVER);
    SetupDiEnumDriverInfo(devInfo, &devInfoData, SPDIT_COMPATDRIVER,
        0, &driverInfoData);

    LOG("Checking driver...\n");

    if (driverInfoData.DriverVersion == 0)
    {
        LOG("FAILED (no driver installed)");
        GnaSelfTestIssue::NO_DRIVER.Handle();
    }
    else
    {
        uint32_t major = driverInfoData.DriverVersion >> 48 & 0xFFFF;

        if (major == 2) {
            LOG("GNA 2.0 driver detected\n");
        }
        else if (major == 10)
        {
            LOG("FAILED (null driver installed)\n");
            LOG("Please install the driver, you may also need to enable the device [see Bring up guide]\n");  //TODO: guide to guide
            GnaSelfTestIssue::NUL_DRIVER.Handle();
        }
        else if (major == 1)
        {
            LOG("GNA 1.0 driver detected\n");
            LOG("Please update the driver\n");
            GnaSelfTestIssue::DRV_1_INSTEAD_2.Handle();
        }
        else
        {
            LOG("FAILED (unknown driver installed)\n");
            GnaSelfTestIssue::UNKNOWN_DRIVER.Handle();
        }

        LOG("Driver version: %d.%d.%d.%d\n",
            int(driverInfoData.DriverVersion >> 48 & 0xFFFF),
            int(driverInfoData.DriverVersion >> 32 & 0xFFFF),
            int(driverInfoData.DriverVersion >> 16 & 0xFFFF),
            int(driverInfoData.DriverVersion & 0xFFFF)
        );
    }

    return 0;
}
