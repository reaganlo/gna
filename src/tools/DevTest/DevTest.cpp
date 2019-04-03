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

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

#include "Script.h"
#include "DeviceController.h"
#include "ModelSetupFactory.h"

#include "gna2-common-api.h"

class ApplicationWrapper
{
public:
    ApplicationWrapper()
    {
    }

    void PrepareScenario()
    {
        script.actions.emplace_back(Action::LoadModel, ModelSetupConvolution_2D, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupConvolution_2D, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupConvolution_2D, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupConvolution_2D, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupMix, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupMix, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMix, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupMix, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupMultibias_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupMultibias_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMultibias_1_1B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupMultibias_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupMultibias_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupMultibias_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMultibias_1_2B, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupMultibias_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupMultibiasPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupMultibiasPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMultibiasPwl_1_1B, 0, 2);
        script.actions.emplace_back(Action::CloseModel, ModelSetupMultibiasPwl_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupMultibiasPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupMultibiasPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMultibiasPwl_1_2B, 0, 3);
        script.actions.emplace_back(Action::CloseModel, ModelSetupMultibiasPwl_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnn_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnn_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnn_1_1B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnn_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnn_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnn_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnn_1_2B, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnn_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAl_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAl_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAl_1_1B, 0, 2);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAl_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAl_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAl_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAl_1_2B, 0, 3);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAl_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnPwl_1_1B, 0, 4);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnPwl_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnPwl_1_2B, 0, 5);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnPwl_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAlPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAlPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAlPwl_1_1B, 0, 6);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAlPwl_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAlPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAlPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAlPwl_1_2B, 0, 7);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAlPwl_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnn_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnn_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnn_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnn_Multibuffer_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnn_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnn_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnn_Multibuffer_2B, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnn_Multibuffer_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAl_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAl_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAl_Multibuffer_1B, 0, 2);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAl_Multibuffer_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAl_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAl_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAl_Multibuffer_2B, 0, 3);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAl_Multibuffer_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnPwl_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnPwl_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnPwl_Multibuffer_1B, 0, 4);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnPwl_Multibuffer_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnPwl_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnPwl_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnPwl_Multibuffer_2B, 0, 5);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnPwl_Multibuffer_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAlPwl_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAlPwl_Multibuffer_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAlPwl_Multibuffer_1B, 0, 6);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAlPwl_Multibuffer_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAlPwl_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnnAlPwl_Multibuffer_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAlPwl_Multibuffer_2B, 0, 7);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAlPwl_Multibuffer_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupSplit_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_1B, 0, 1);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_1B, 0, 1);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_1B, 1, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_1B, 1, 0);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_1B, 1, 1);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_1B, 1, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupSplit_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupSplit_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 0, 1);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 0, 1);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 1, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 1, 0);
        script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 1, 1);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 1, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupSplit_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupCopy_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupCopy_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupCopy_1, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupCopy_1, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupCopy_2, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupCopy_2, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupCopy_2, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupCopy_2, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupCopy_3, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupCopy_3, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupCopy_3, 0, 2);
        script.actions.emplace_back(Action::CloseModel, ModelSetupCopy_3, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupCopy_4, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupCopy_4, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupCopy_4, 0, 3);
        script.actions.emplace_back(Action::CloseModel, ModelSetupCopy_4, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupTranspose_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupTranspose_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupTranspose_1, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupTranspose_1, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupTranspose_2, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupTranspose_2, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupTranspose_2, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupTranspose_2, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDiagonal_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDiagonal_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDiagonal_1_1B, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDiagonal_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDiagonal_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDiagonal_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDiagonal_1_2B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDiagonal_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDiagonalPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDiagonalPwl_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDiagonalPwl_1_1B, 0, 2);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDiagonalPwl_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDiagonalPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDiagonalPwl_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDiagonalPwl_1_2B, 0, 3);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDiagonalPwl_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupRecurrent_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupRecurrent_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupRecurrent_1_2B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupRecurrent_1_2B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupRecurrent_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupRecurrent_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupRecurrent_1_1B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupRecurrent_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupConvolution_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupConvolution_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupConvolution_1, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupConvolution_1, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupConvolutionPwl_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupConvolutionPwl_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupConvolutionPwl_1, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupConvolutionPwl_1, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupPooling_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupPooling_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupPooling_1, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupPooling_1, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupGmm_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupGmm_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupGmm_1, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupGmm_1, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupGmmAl_1, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupGmmAl_1, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupGmmAl_1, 0, 1);
        script.actions.emplace_back(Action::CloseModel, ModelSetupGmmAl_1, 0, 0);
    }

    void PrepareModels()
    {
        // TODO single model scenario support only
        /*ModelSetupFactory mf(deviceController);
        auto model0 = mf.CreateModel(0);
        auto model1 = mf.CreateModel(0);

        script.models.push_back(std::move(model0));
        script.models.push_back(std::move(model1));*/
    }

    void Run()
    {
        ModelSetupFactory msf(deviceController);
        IModelSetup::UniquePtr modelSetup;
        for (auto action = script.actions.begin(); action != script.actions.end(); ++action)
        {
            // TODO single model scenario support only
            //auto model = script.models[action->modelIndex].get();

            switch (action->actionType)
            {
            case Action::LoadModel:
                if (modelSetup)
                {
                    throw std::runtime_error("Script error: model already loaded");
                }

                modelSetup = msf.CreateModel(action->modelSetup);
                break;

            case Action::CloseModel:
                if (!modelSetup)
                {
                    throw std::runtime_error("Script error: no model to close");
                }

                modelSetup.reset();
                break;

            case Action::Score:
            {
                auto config = modelSetup->ConfigId(action->modelIndex, action->configIndex);
                deviceController.RequestSetAcceleration(config, gna_acceleration::GNA_AUTO);
                deviceController.RequestSetConsistency(config, Gna2DeviceVersionAlderLake);
                gna_request_id requestId;
                deviceController.RequestEnqueue(config, &requestId);
                deviceController.RequestWait(requestId);
                break;
            }
            case Action::CheckReferenceOutput:
                modelSetup->checkReferenceOutput(action->modelIndex, action->configIndex);
                std::cout << "Test passed" << std::endl;
                break;
            }
        }
    }

    void CloseScenario()
    {
    }

    ~ApplicationWrapper()
    {
    }

private:

    DeviceController deviceController;

    std::vector<ActionScript> scripts;
    ActionScript script;
};

int main()
{
    const int retCode = 0;
    const int retError = -1;

    try
    {
        ApplicationWrapper app;
        app.PrepareScenario();
        app.PrepareModels();
        app.Run();
        app.CloseScenario();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return retError;
    }
    catch (...)
    {
        std::cerr << "Unknown exception" << std::endl;
        return retError;
    }
    return retCode;
}
