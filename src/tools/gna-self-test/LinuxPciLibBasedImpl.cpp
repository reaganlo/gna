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
extern "C"{
#include <pci/pci.h>
}

std::vector<PciDeviceInfo> LinuxGnaSelfTestHardwareStatus::getDevicesList()
{
    logger.Verbose("INFO in getDevicesList LIBPCI method\n");
    std::vector<PciDeviceInfo> devList;
    struct pci_access *pciCtrl;
    struct pci_dev *dev;
    char devNameBuf[1024], *devName;
    pciCtrl = pci_alloc();     //PCI library control structure allocation
    pci_init(pciCtrl);         //PCI library control structure initialization
    pci_scan_bus(pciCtrl);     //Get the list of devices
    for (dev=pciCtrl->devices; dev; dev=dev->next)
    {
        pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS);
        devName = pci_lookup_name(pciCtrl, devNameBuf, sizeof(devNameBuf), PCI_LOOKUP_DEVICE, dev->vendor_id, dev->device_id);
        PciDeviceInfo di;
        di.vendorId = dev->vendor_id;
        di.deviceId = dev->device_id;
        di.devClass = dev->device_class;
        di.irq = dev->irq;
        di.domain = dev->domain;
        di.bus = dev->bus;
        di.dev = dev->dev;
        di.func = dev->func;
        di.irqPin = pci_read_byte(dev, PCI_INTERRUPT_PIN);
        di.name = devName;
        devList.push_back(di);
    }
    pci_cleanup(pciCtrl);
    return devList;
}
