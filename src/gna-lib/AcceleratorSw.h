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

#include "Accelerator.h"
#include "KernelDispatcher.h"
#include "GmmLayer.h"

namespace GNA
{

class AcceleratorSw : public Accelerator
{
friend class AcceleratorCnl;
public:    
    /** @See IAccelerator */
    status_t Wait(
        Request*        r,
        uint32_t        timeout,
        perf_t*         perfResults)
        override;

    /**
     * Initializes processing device if available
     *
     * @nProcessorType      acceleration mode
     * @status      (out)   status of opening device
     */
    AcceleratorSw(acceleration nProcessorType);

    status_t init()
        override;

protected:
    /**
     * software calculation kernels dispatcher
     */
    KernelDispatcher    kd;

    /** @See Accelerator */
    status_t submit(
         Request*   r)
         override;

private:
    status_t gmmSoftwareKernel(
        GmmLayer* gmm,
        req_profiler* profiler);

    status_t xnnSoftwareKernel(
        SoftwareModel* model,
        req_profiler* profiler);

    static inline status_t checkScoresSaturation(
        uint32_t nGMMs,
        uint32_t nVectors,
        uint32_t *pS,
        uint32_t maxScore);

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    AcceleratorSw() = delete;
    AcceleratorSw(const AcceleratorSw &) = delete;
    AcceleratorSw& operator=(const AcceleratorSw&) = delete;
};

}
