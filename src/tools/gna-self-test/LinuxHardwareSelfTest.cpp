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
#include "gna.h"

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
    LOG("WARNING checking the GNA device in Linux/Android is not supported\n");
    auto devList = getDevicesList();
    for(auto dev : devList)
    {
        for(auto known: knownGNADevsIds)
        {
            if(known.first.first == dev.vendorId && known.first.second == dev.deviceId)
            {
                devFound=true;
                LOG("SUCCESS the device have been found on Your system\n");
                LOG("Device name: < %s >\n",known.second.c_str());
                LOG("Details: < %s >\n",dev.toString().c_str());
            }
        }
    }
    return devFound?0:-1;
}

int LinuxGnaSelfTestHardwareStatus::checkDriver()
{
    LOG("WARNING checking the GNA driver in Linux/Android is not supported\n");

    readCmdOutput(GNA_ST_LSMOD);
    readCmdOutput(GNA_ST_MODPROBE);

    LOG("INFO Looking for device node /dev/gna[0-%d]\n",DEFAULT_GNA_DEV_NODE_RANGE-1);
    std::string devEntry = devfsGnaNode(DEFAULT_GNA_DEV_NODE_RANGE);
    if(devEntry.size()==0)
    {
        LOG("WARNING Looks like there is no node at /dev/gna[0-%d]\n",DEFAULT_GNA_DEV_NODE_RANGE-1);
        return -1;
    }
    LOG("INFO GNA Node at <%s> has been found\n",devEntry.c_str());
    return 0;
}

std::string LinuxGnaSelfTestHardwareStatus::devfsGnaNode(int range)
{
    int fd;
    std::string found;
    struct gna_capabilities gnaCaps;
    for(int i = 0; i < range; i++)
    {
        char name[12];
        sprintf(name, "/dev/gna%d", i);

        fd = open(name, O_RDWR);
        if(fd == -1)
        {
            continue;
        }
        if(0 == ioctl(fd, GNA_CPBLTS, &gnaCaps))
        {
            LOG("INFO GNA device with id = %d found\n",(int)gnaCaps.device_type);
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
    LOG("INFO running <%s>\n",command);
    FILE * file = popen(command,"r");
    if(file!=NULL)
    {
        char line[256];
        LOG("<%s>: --------OUTPUT------------------\n",command);
        do
        {
            char* str = fgets(line,sizeof(line),file);
            if(str!=NULL)
            {
                output+=str;
                line[strcspn(str,"\r\n")]=0;
                LOG("<%s>: <%s>\n",command,str);
            }
            else
            {
                LOG("<%s>: --------END-OF-OUTPUT-----------\n",command);
                break;
            }
        }
        while (true);
        pclose(file);
    }
    else
    {
        LOG("ERROR Can not read output from <%s>\n",command);
    }
    return output;
}
void LinuxGnaSelfTestHardwareStatus::determineUserIdentity() const
{
    uid_t uid = getuid();
    uid_t euid = geteuid();
    LOG("UID: %d; EUID: %d\n",(int)uid,(int)euid);
    if(euid!=0)
    {
        LOG("NOTICE Consider running as Administrator [sudo gna-self-test]\n");
    }
}
