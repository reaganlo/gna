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

#ifndef __GNA2_API_WRAPPER_H
#define __GNA2_API_WRAPPER_H

#include "gna2-common-impl.h"

#include "GnaException.h"

#include <functional>
#include <stdexcept>
#include <stdint.h>
#include <algorithm>

namespace GNA
{

class ApiWrapper
{
public:

    template<typename ReturnType = ApiStatus, typename ... ErrorValueType>
    static ReturnType ExecuteSafely(const std::function<ReturnType()>& command,
        ErrorValueType... error)
    {
        try
        {
            return command();
        }
        catch (const GnaModelException &e)
        {
            e.Print();
            return ReturnError<ReturnType>(error...);
        }
        catch (const GnaException &e)
        {
            e.Print();
            return ReturnError<ReturnType>(error...);
        }
        catch (const std::exception& e)
        {
            LogException(e);
            return ReturnError<ReturnType>(error...);
        }
    }

private:
    template<typename ReturnType, typename... ErrorValueType>
    static ReturnType ReturnError(ErrorValueType... returnValues)
    {
        ReturnType returnValueContainer[1] = {returnValues...};
        return returnValueContainer[0];
    }

    template<typename ExceptionType>
    static void LogException(const ExceptionType& e);
};

template<>
    inline ApiStatus ApiWrapper::ReturnError(const GnaModelException &e)
    {
        return e.GetStatus();
    }

    template<>
    inline ApiStatus ApiWrapper::ReturnError(const GnaException &e)
    {
        return e.GetStatus();
    }

    template<>
    inline ApiStatus ApiWrapper::ReturnError()
    {
        return Gna2StatusUnknownError;
    }

    template<>
    inline void ApiWrapper::ReturnError()
    {
        return;
    }

    template<>
    inline void ApiWrapper::LogException(const std::exception& e)
    {
        Log->Error("Unknown error: %s.\n", e.what());
    }
}

#endif //ifndef __GNA2_API_WRAPPER_H
