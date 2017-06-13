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

#include "IModelSetup.h"
#include "DeviceController.h"

class SetupTransposeModel : public IModelSetup
{
public:
    gna_model_id ModelId() const override
    {
        return 0;
    }

    gna_request_cfg_id ConfigId(int /*index*/) const override
    {
        // this model has only one Request Configuration
        return configId;
    }

    SetupTransposeModel(DeviceController & deviceCtrl);

    ~SetupTransposeModel();

    void checkReferenceOutput() const override;

private:
    void sampleTransposeLayer();

    DeviceController & deviceController;

    gna_model_id modelId;
    gna_request_cfg_id configId;

    intel_nnet_type_t nnet;
    intel_pwl_func_t pwl;
    intel_nnet_layer_t transposeLayer;

    void * inputBuffer = nullptr;
    void * outputBuffer = nullptr;
};
