/*
 INTEL CONFIDENTIAL
 Copyright 2018 Intel Corporation.

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

#pragma once

#include "IoctlSender.h"

#include "gna.h"

#include <map>
#include <memory>

#include "common.h"

#include "HardwareRequest.h"
#include "Request.h"
#include "Validator.h"

namespace GNA
{

class LinuxIoctlSender : public IoctlSender
{
public:
    LinuxIoctlSender() = default;

    virtual void Open() override;

    virtual void IoctlSend(const GnaIoctlCommand command, void * const inbuf, const uint32_t inlen, void * const outbuf, const uint32_t outlen) override;

    virtual GnaCapabilities GetDeviceCapabilities() const override;

    virtual uint64_t MemoryMap(void *memory, size_t memorySize) override;

    virtual void MemoryUnmap(uint64_t memoryId) override;

    virtual RequestResult Submit(HardwareRequest * const hardwareRequest, RequestProfiler * const profiler) override;

private:
    LinuxIoctlSender(const LinuxIoctlSender &) = delete;
    LinuxIoctlSender& operator=(const LinuxIoctlSender&) = delete;

    void createRequestDescriptor(HardwareRequest *hardwareRequest);

    status_t parseHwStatus(__u32 hwStatus) const;

    int gnaFileDescriptor = -1;
    GnaCapabilities deviceCapabilities;

    std::unique_ptr<gna_score_cfg> scoreConfig = nullptr;
    size_t scoreConfigSize = 0;
};

}
