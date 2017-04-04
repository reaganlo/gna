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

#include "IAccelerator.h"
#include "gmm.h"
#include "GmmLayer.h"
#include "Request.h"
#include "XnnKernelApi.h"

namespace GNA
{

class AcceleratorSw : public IAccelerator
{
public:    
    AcceleratorSw(acceleration accel);

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    AcceleratorSw() = delete;
    AcceleratorSw(const AcceleratorSw &) = delete;
    AcceleratorSw& operator=(const AcceleratorSw&) = delete;

    status_t Score(
        const CompiledModel& model,
        const RequestConfiguration& requestConfiguration,
              RequestProfiler *profiler,
              KernelBuffers *buffers) override;

    status_t Score(
        const CompiledModel& model,
        const SubModel& submodel,
        const RequestConfiguration& requestConfiguration,
              RequestProfiler *profiler,
              KernelBuffers *buffers) override;

protected:
    XnnKernel *xnnKernel;
    GmmKernel *gmmKernel;

private:
    void applyRequestBuffersToLayer(
        const LayerConfiguration * const layerConfiguration,
        const Layer * const layer,
        nn_layer * const sourceLayer,
        uint32_t &nOuts,
        const uint32_t * &activeIndices);

    void gmmSoftwareKernel(
        GmmLayer* gmm,
        const LayerConfiguration * const layerConfiguration,
        uint32_t* const nSaturated);

    static inline void checkScoresSaturation(
        uint32_t nGMMs,
        uint32_t nVectors,
        const uint32_t * pS,
        uint32_t maxScore,
        uint32_t* const nSaturated);
};

}
