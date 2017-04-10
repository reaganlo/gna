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

#include "ModelCompiler.h"

#include "gna-api.h"
#include "FakeDetector.h"
#include "Validator.h"

using std::make_unique;

using namespace GNA;

const size_t ModelCompiler::MaximumInternalModelSize = CalculateInternalModelSize(XNN_LAYERS_MAX_COUNT, GMM_LAYERS_MAX_COUNT);

const size_t ModelCompiler::CalculateModelSize(const size_t requestedSize, const uint16_t layerCount,
    const uint16_t gmmCount)
{
    auto internalSize = CalculateInternalModelSize(layerCount, gmmCount);
    auto totalSize = requestedSize + internalSize;
    Expect::InRange(totalSize, 1, 256*1024*1024, GNA_INVALIDMEMSIZE);
    return totalSize;
}

const size_t ModelCompiler::CalculateInternalModelSize(const uint16_t layerCount,
    const uint16_t gmmCount)
{
    // TODO:INTEGRATION: add detector reference to c-tor and calculate hardware size if applicable
    // for model dumper use fake detector in device
    return HardwareModel::CalculateDescriptorSize(layerCount, gmmCount);
}

void ModelCompiler::CascadeCompile(CompiledModel &model, const AccelerationDetector& detector)
{
    model.CompileSoftwareModel();
    if (detector.IsHardwarePresent())
    {
        model.CompileHardwareModel(detector);
    }

    model.CreateSubmodels(detector);
}
