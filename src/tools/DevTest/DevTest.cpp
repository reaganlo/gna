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

#include "Script.h"
#include "DeviceController.h"
#include "ModelSetupFactory.h"

class ApplicationWrapper
{
public:
    ApplicationWrapper()
    {
    }

    void PrepareScenario()
    {
        //script.actions.emplace_back(Action::LoadModel, ModelSetupMix, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupMix, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMix, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupMix, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupMultibias_1_2B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupMultibias_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMultibias_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupMultibias_1_2B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupMultibias_1_1B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupMultibias_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupMultibias_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupMultibias_1_1B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupDnnAl_1_1B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupDnnAl_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnnAl_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupDnnAl_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupDnn_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupDnn_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDnn_1_2B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupDnn_1_2B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupSplit_1_2B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 0, 1);
        //script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 1, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupSplit_1_2B, 1, 1);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 0, 1);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 1, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupSplit_1_2B, 0, 1);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupSplit_1_2B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupCopy_1, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupCopy_1, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupCopy_1, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupCopy_1, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupTranspose_1, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupTranspose_1, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupTranspose_1, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupTranspose_1, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupDiagonal_1_2B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupDiagonal_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDiagonal_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupDiagonal_1_2B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupDiagonal_1_1B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupDiagonal_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupDiagonal_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupDiagonal_1_1B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupRecurrent_1_2B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupRecurrent_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupRecurrent_1_2B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupRecurrent_1_2B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupRecurrent_1_1B, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupRecurrent_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupRecurrent_1_1B, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupRecurrent_1_1B, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupConvolution_1, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupConvolution_1, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupConvolution_1, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupConvolution_1, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupPooling_1, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupPooling_1, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupPooling_1, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupPooling_1, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupGmm_1, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupGmm_1, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupGmm_1, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupGmm_1, 0, 0);

        //script.actions.emplace_back(Action::LoadModel, ModelSetupGmmAl_1, 0, 0);
        //script.actions.emplace_back(Action::Score, ModelSetupGmmAl_1, 0, 0);
        //script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupGmmAl_1, 0, 0);
        //script.actions.emplace_back(Action::CloseModel, ModelSetupGmmAl_1, 0, 0);
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
                gna_request_id requestId;
                deviceController.RequestEnqueue(modelSetup->ConfigId(action->modelIndex, action->configIndex), GNA_AUTO, &requestId);
                GnaRequestWait(requestId, 5*60*1000);
                break;

            case Action::CheckReferenceOutput:
                modelSetup->checkReferenceOutput(action->modelIndex, action->configIndex);
                break;
            }
        }
    }

    void CloseScenario()
    {
    }

    ~ApplicationWrapper()
    {
        printError();
    }

private:
    void printError(int error = 0)
    {
        if (0 != error)
        {
        }
    }

    DeviceController deviceController;

    std::vector<ActionScript> scripts;
    ActionScript script;
};

static int handleException();

int main()
{
    int retCode = 0;

    try
    {
        ApplicationWrapper app;
        app.PrepareScenario();
        app.PrepareModels();
        app.Run();
        app.CloseScenario();
    }
    catch (...)
    {
        retCode = handleException();
    }

    return retCode;
}

static int handleException()
{
    int retCode = -1;

    std::cout << "\nException!!!\n";
    try
    {
        throw;
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception\n";
    }

    return retCode;
}
