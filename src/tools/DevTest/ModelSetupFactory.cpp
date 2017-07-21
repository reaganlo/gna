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

#include "ModelSetupFactory.h"
#include "SetupConvolutionModel.h"
#include "SetupCopyModel.h"
#include "SetupDiagonalModel.h"
#include "SetupGmmModel.h"
#include "SetupMixModel.h"
#include "SetupDnnModel_1.h"
#include "SetupMultibiasModel_1.h"
#include "SetupPoolingModel.h"
#include "SetupRecurrentModel.h"
#include "SetupTransposeModel.h"
#include "SetupSplitModel.h"

IModelSetup::UniquePtr ModelSetupFactory::CreateModel(ModelSetupType ms)
{
    IModelSetup::UniquePtr ptr;

    switch (ms)
    {
    default:
    case ModelSetupDnn_1_1B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, false, false, false);
        break;
    case ModelSetupDnn_1_2B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, true, false, false);
        break;
    case ModelSetupDnnAl_1_1B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, false, true, false);
        break;
    case ModelSetupDnnAl_1_2B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, true, true, false);
        break;
    case ModelSetupDnnPwl_1_1B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, false, false, true);
        break;
    case ModelSetupDnnPwl_1_2B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, true, false, true);
        break;
    case ModelSetupDnnAlPwl_1_1B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, false, true, true);
        break;
    case ModelSetupDnnAlPwl_1_2B:
        ptr = std::make_unique<SetupDnnModel_1>(deviceController, true, true, true);
        break;
    case ModelSetupMultibias_1_1B:
        ptr = std::make_unique<SetupMultibiasModel_1>(deviceController, false, false);
        break;
    case ModelSetupMultibias_1_2B:
        ptr = std::make_unique<SetupMultibiasModel_1>(deviceController, true, false);
        break;
    case ModelSetupMultibiasPwl_1_1B:
        ptr = std::make_unique<SetupMultibiasModel_1>(deviceController, false, true);
        break;
    case ModelSetupMultibiasPwl_1_2B:
        ptr = std::make_unique<SetupMultibiasModel_1>(deviceController, true, true);
        break;
    case ModelSetupConvolution_1:
        ptr = std::make_unique<SetupConvolutionModel>(deviceController, false);
        break;
    case ModelSetupConvolutionPwl_1:
        ptr = std::make_unique<SetupConvolutionModel>(deviceController, true);
        break;
    case ModelSetupPooling_1:
        ptr = std::make_unique<SetupPoolingModel>(deviceController);
        break;
    case ModelSetupRecurrent_1_1B:
        ptr = std::make_unique<SetupRecurrentModel>(deviceController, false);
        break;
    case ModelSetupRecurrent_1_2B:
        ptr = std::make_unique<SetupRecurrentModel>(deviceController, true);
        break;
    case ModelSetupDiagonal_1_1B:
        ptr = std::make_unique<SetupDiagonalModel>(deviceController, false, false);
        break;
    case ModelSetupDiagonal_1_2B:
        ptr = std::make_unique<SetupDiagonalModel>(deviceController, true, false);
        break;
    case ModelSetupDiagonalPwl_1_1B:
        ptr = std::make_unique<SetupDiagonalModel>(deviceController, false, true);
        break;
    case ModelSetupDiagonalPwl_1_2B:
        ptr = std::make_unique<SetupDiagonalModel>(deviceController, true, true);
        break;
    case ModelSetupCopy_1:
        ptr = std::make_unique<SetupCopyModel>(deviceController);
        break;
    case ModelSetupTranspose_1:
        ptr = std::make_unique<SetupTransposeModel>(deviceController);
        break;
    case ModelSetupGmm_1:
        ptr = std::make_unique<SetupGmmModel>(deviceController, false);
        break;
    case ModelSetupGmmAl_1:
        ptr = std::make_unique<SetupGmmModel>(deviceController, true);
        break;
    case ModelSetupMix:
        ptr = std::make_unique<SetupMixModel>(deviceController);
        break;
    case ModelSetupSplit_1_2B:
        ptr = std::make_unique<SetupSplitModel>(deviceController, true, false, false);
        break;
    }
    return ptr;
}
