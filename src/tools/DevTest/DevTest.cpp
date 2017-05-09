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
        script.actions.emplace_back(Action::LoadModel, ModelSetupBasic_1_1B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupBasic_1_1B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupBasic_1_1B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupBasic_1_1B, 0, 0);

        script.actions.emplace_back(Action::LoadModel, ModelSetupBasic_1_2B, 0, 0);
        script.actions.emplace_back(Action::Score, ModelSetupBasic_1_2B, 0, 0);
        script.actions.emplace_back(Action::CheckReferenceOutput, ModelSetupBasic_1_2B, 0, 0);
        script.actions.emplace_back(Action::CloseModel, ModelSetupBasic_1_2B, 0, 0);
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
                    throw std::exception("Script error: model already loaded");
                }

                modelSetup = msf.CreateModel(action->modelSetup);
                break;

            case Action::CloseModel:
                if (!modelSetup)
                {
                    throw std::exception("Script error: no model to close");
                }

                modelSetup.reset();
                break;

            case Action::Score:
                gna_request_id requestId;
                deviceController.RequestEnqueue(modelSetup->ConfigId(action->configIndex), GNA_SOFTWARE, &requestId);
                GnaRequestWait(requestId, 100);
                break;

            case Action::CheckReferenceOutput:
                modelSetup->checkReferenceOutput();
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
