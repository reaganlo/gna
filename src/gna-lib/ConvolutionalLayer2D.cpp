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

#include "ConvolutionalLayer2D.h"

#include "AccelerationDetector.h"
#include "LayerConfiguration.h"
#include "Expect.h"
#include "HardwareLayer.h"

using namespace GNA;

ConvolutionalLayer2D::ConvolutionalLayer2D(nn_layer const * const layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {ConvolutionalTransform2D, ActivationTransform, PoolingTransform2D}, BaseAddress())
{
    Expect::One(Input.at(GNA_DIM_N), XNN_ERR_GROUPING);
    Expect::One(Output.at(GNA_DIM_N), XNN_ERR_GROUPING);
    Expect::Equal(Output.Size, GetOutputTransform()->Output->Size, status_t::XNN_ERR_OUTPUT_VOLUME);

    // performed for layer size validation
    HardwareLayerCnn2D::GetKernelWorkGroupSize(
        AccelerationDetector::GetDeviceVersion(validator->Device),
        Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D),
        Transforms.Get<PoolingFunction2D>(PoolingTransform2D),
        GetOutputTransform()->Output->Mode);

    Layer::ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(nullptr, accel, executionConfig); };

    Layer::Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(&layerConfiguration, accel, executionConfig); };

}
