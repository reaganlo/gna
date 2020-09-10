/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

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

#include "Layer.h"
#include "XnnKernel.h"

namespace GNA
{

// TODO:3: Refactor to use tensors and functions
class CopyLayer : public Layer
{
public:
    CopyLayer(const Gna2Operation& operation, const BaseValidator& validatorIn);

    virtual ~CopyLayer() = default;
    void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    const uint32_t ColumnCount;
    const uint32_t RowCount;

protected:
    virtual DataConfig GetDataMode() const override;

    static Shape GetCopyShape(const Gna2Operation& operation);

private:
    void computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const;
    void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const;

    const KernelMap<CopyKernel>& copyKernels;
    CopyConfig copyHiddenConfig;
    static const FullCapabilitiesMap limits;
};

}
