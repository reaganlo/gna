/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

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

#include <string>

#include "IAccelerator.h"
#include "IoctlSender.h"
#include "Request.h"

namespace GNA
{

class AcceleratorHw : public IAccelerator, protected IoctlSender
{
public:
    using IAccelerator::IAccelerator;

    ~AcceleratorHw() {};

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    AcceleratorHw() = delete;
    AcceleratorHw(const AcceleratorHw &) = delete;
    AcceleratorHw& operator=(const AcceleratorHw&) = delete;

    status_t Score(
        const CompiledModel& model,
        const RequestConfiguration& requestConfiguration,
              req_profiler *profiler,
              aligned_fv_bufs *buffers) override;

    status_t Score(
        const CompiledModel& model,
        const SubModel& submodel,
        const RequestConfiguration& requestConfiguration,
              req_profiler *profiler,
              aligned_fv_bufs *buffers) override;

protected:
    bool driverDebug = false;
    /**
     * Internal hw input buffer size in KB
     */
    uint8_t    hwInBuffSize = 0;

    /**
     * virtual hw verification methods implemented in HW VERBOSE version only
     * NOTE: DO NOT USE WITH MULTITHREADED APPLICATIONS
     */

    /**
     * Performs hw verification in HW VERBOSE version only
     * NOTE: DO NOT USE WITH MULTITHREADED APPLICATIONS
     *
     * @path    path to set-register file with verification commands
     * @return  true if valid set-register file  was found
     */
    virtual bool SetRegister(std::string path);

    virtual bool SetConfig(std::string path, hw_calc_in_t* inData);

    virtual bool SetDescriptor(std::string path, XNN_LYR* buff, hw_calc_in_t* inData);

    virtual void HwVerifier(Request* r);

    // TODO: replace SoftwareModel with compiled model
    virtual void HwVerifier(SoftwareModel* model, status_t scoring_status);
};

}
