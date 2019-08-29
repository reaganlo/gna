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
#include <map>
#include <string>
#include <vector>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <cstring>
#include "gna-h-wrapper.h"

static const std::map<std::pair<u_int16_t,u_int16_t>, std::string> knownGNADevsIds
{
    {{0x8086,0x5A11}, "GNA 0.9 CNL"},
    {{0x8086,0x3190}, "GNA 1.0 GLK"},
    {{0x8086,0x4511}, "GNA 1.0 EHL"},
    {{0x8086,0x8A11}, "GNA 1.0 ICL"},
    {{0x8086,0x9A11}, "GNA 2.0 TGL"}
//  uncomment for testing purposes then gns-self-test will discover these devices as GNAs
//  {{0x10ec,0x8168}, "r8169 pseudo GNA device - for testing"}, // Realtek's 8169 Ethernet Controller
//  {{0x8086,0x2668}, "Android emulator's pseudo GNA device - for testing"}, // Some random Android emulator's device
};

void LinuxGnaSelfTestHardwareStatus::initHardwareInfo()
{
    hardwareAvailable = (checkHWId() == 0);
}

void LinuxGnaSelfTestHardwareStatus::initDriverInfo()
{
    driverAvailable = (checkDriver() == 0);
}

int LinuxGnaSelfTestHardwareStatus::checkHWId()
{
    bool devFound=false;
    auto devList = getDevicesList();
    for(auto dev : devList)
    {
        for(auto known: knownGNADevsIds)
        {
            if(known.first.first == dev.vendorId && known.first.second == dev.deviceId)
            {
                devFound=true;
                logger.Verbose("SUCCESS the device have been found on Your system\n");
                logger.Verbose("Device name: < %s >\n",known.second.c_str());
                logger.Verbose("Details: < %s >\n",dev.toString().c_str());
            }
        }
    }
    return devFound?0:-1;
}

int LinuxGnaSelfTestHardwareStatus::checkDriver()
{
    readCmdOutput(GNA_ST_LSMOD);
    readCmdOutput(GNA_ST_MODPROBE);

    logger.Verbose("Looking for device node /dev/gna[0-%d]\n",DEFAULT_GNA_DEV_NODE_RANGE-1);
    std::string devEntry = devfsGnaNode(DEFAULT_GNA_DEV_NODE_RANGE);
    if(devEntry.empty())
    {
        GnaSelfTestLogger::Error("ERROR Looks like there is no node at /dev/gna[0-%d]\n",DEFAULT_GNA_DEV_NODE_RANGE-1);
        return -1;
    }
    logger.Verbose("INFO GNA Node at <%s> has been found\n",devEntry.c_str());
    return 0;
}

std::string LinuxGnaSelfTestHardwareStatus::devfsGnaNode(uint8_t range)
{
    int fd;
    std::string found;
    struct gna_getparam params[3] =
    {
        { GNA_PARAM_DEVICE_ID, 0 },
        { GNA_PARAM_INPUT_BUFFER_S , 0 },
        { GNA_PARAM_RECOVERY_TIMEOUT, 0 },
    };

    for(uint8_t i = 0; i < range; i++)
    {
        auto name = "/dev/gna" + std::to_string(i);
        fd = open(name.c_str(), O_RDWR);
        if(fd == -1)
        {
            continue;
        }
        if(ioctl(fd, GNA_IOCTL_GETPARAM, &params[0]) == 0
            && ioctl(fd, GNA_IOCTL_GETPARAM, &params[1]) == 0
            && ioctl(fd, GNA_IOCTL_GETPARAM, &params[2]) == 0)
        {
            logger.Verbose("INFO GNA device of type = %llX found\n", params[0].value);
            found = name;
            close(fd);
            break;
        }
        close(fd);
    }
    return found;
}

std::string LinuxGnaSelfTestHardwareStatus::readCmdOutput(const char* command) const
{
    std::string output;
    logger.Verbose("INFO running <%s>\n",command);
    FILE * file = popen(command,"r");
    if(file!=NULL)
    {
        char line[256];
        logger.Verbose("<%s>: --------OUTPUT------------------\n",command);
        do
        {
            char* str = fgets(line,sizeof(line),file);
            if(str!=NULL)
            {
                output+=str;
                line[strcspn(str,"\r\n")]=0;
                logger.Verbose("<%s>: <%s>\n",command,str);
            }
            else
            {
                logger.Verbose("<%s>: --------END-OF-OUTPUT-----------\n",command);
                break;
            }
        }
        while (true);
        pclose(file);
    }
    else
    {
        GnaSelfTestLogger::Error("ERROR Can not read output from <%s>\n",command);
    }
    return output;
}
void LinuxGnaSelfTestHardwareStatus::determineUserIdentity() const
{
    uid_t uid = getuid();
    uid_t euid = geteuid();
    logger.Verbose("UID: %d; EUID: %d\n",(int)uid,(int)euid);
    if(euid!=0)
    {
        logger.Verbose("Consider running as Administrator [sudo gna-self-test]\n");
    }
}
