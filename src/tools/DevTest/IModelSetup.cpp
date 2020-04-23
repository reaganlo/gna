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

#include "IModelSetup.h"

#include <map>
#include <string>

std::string GetModelSetupTypeStr(ModelSetupType type)
{
    static std::map<ModelSetupType, std::string> modelSetupTypeMap{
    {ModelSetupDnn_1_1B, "ModelSetupDnn_1_1B"},
    {ModelSetupDnn_1_2B, "ModelSetupDnn_1_2B"},
    {ModelSetupDnnAl_1_1B, "ModelSetupDnnAl_1_1B"},
    {ModelSetupDnnAl_1_2B, "ModelSetupDnnAl_1_2B"},
    {ModelSetupDnnPwl_1_1B, "ModelSetupDnnPwl_1_1B"},
    {ModelSetupDnnPwl_1_2B, "ModelSetupDnnPwl_1_2B"},
    {ModelSetupDnnAlPwl_1_1B, "ModelSetupDnnAlPwl_1_1B"},
    {ModelSetupDnnAlPwl_1_2B, "ModelSetupDnnAlPwl_1_2B"},

    {ModelSetupDnn_Multibuffer_1B, "ModelSetupDnn_Multibuffer_1B"},
    {ModelSetupDnn_Multibuffer_2B, "ModelSetupDnn_Multibuffer_2B"},
    {ModelSetupDnnAl_Multibuffer_1B, "ModelSetupDnnAl_Multibuffer_1B"},
    {ModelSetupDnnAl_Multibuffer_2B, "ModelSetupDnnAl_Multibuffer_2B"},
    {ModelSetupDnnPwl_Multibuffer_1B, "ModelSetupDnnPwl_Multibuffer_1B"},
    {ModelSetupDnnPwl_Multibuffer_2B, "ModelSetupDnnPwl_Multibuffer_2B"},
    {ModelSetupDnnAlPwl_Multibuffer_1B, "ModelSetupDnnAlPwl_Multibuffer_1B"},
    {ModelSetupDnnAlPwl_Multibuffer_2B, "ModelSetupDnnAlPwl_Multibuffer_2B"},

    {ModelSetupMultibias_1_1B, "ModelSetupMultibias_1_1B"},
    {ModelSetupMultibias_1_2B, "ModelSetupMultibias_1_2B"},
    {ModelSetupMultibiasAl_1_1B, "ModelSetupMultibiasAl_1_1B"},
    {ModelSetupMultibiasAl_1_2B, "ModelSetupMultibiasAl_1_2B"},
    {ModelSetupMultibiasPwl_1_1B, "ModelSetupMultibiasPwl_1_1B"},
    {ModelSetupMultibiasPwl_1_2B, "ModelSetupMultibiasPwl_1_2B"},
    {ModelSetupMultibiasAlPwl_1_1B, "ModelSetupMultibiasAlPwl_1_1B"},
    {ModelSetupMultibiasAlPwl_1_2B, "ModelSetupMultibiasAlPwl_1_2B"},

    {ModelSetupRecurrent_1_1B, "ModelSetupRecurrent_1_1B"},
    {ModelSetupRecurrent_1_2B, "ModelSetupRecurrent_1_2B"},

    {ModelSetupDiagonal_1_1B, "ModelSetupDiagonal_1_1B"},
    {ModelSetupDiagonal_1_2B, "ModelSetupDiagonal_1_2B"},
    {ModelSetupDiagonalPwl_1_1B, "ModelSetupDiagonalPwl_1_1B"},
    {ModelSetupDiagonalPwl_1_2B, "ModelSetupDiagonalPwl_1_2B"},

    {ModelSetupConvolution_1, "ModelSetupConvolution_1"},
    {ModelSetupConvolutionPwl_1, "ModelSetupConvolutionPwl_1"},

    {ModelSetupPooling_1, "ModelSetupPooling_1"},

    {ModelSetupGmm_1, "ModelSetupGmm_1"},
    {ModelSetupGmmAl_1, "ModelSetupGmmAl_1"},

    {ModelSetupCopy_1, "ModelSetupCopy_1"},
    {ModelSetupCopy_2, "ModelSetupCopy_2"},
    {ModelSetupCopy_3, "ModelSetupCopy_3"},
    {ModelSetupCopy_4, "ModelSetupCopy_4"},

    {ModelSetupTranspose_1, "ModelSetupTranspose_1"},
    {ModelSetupTranspose_2, "ModelSetupTranspose_2"},

    {ModelSetupMix, "ModelSetupMix,"},
    {ModelSetupSplit_1_1B, "ModelSetupSplit_1_1B"},
    {ModelSetupSplit_1_2B, "ModelSetupSplit_1_2B"},

    {ModelSetupConvolution_2D, "ModelSetupConvolution_2D"},
    };
    return modelSetupTypeMap.at(type);
}
