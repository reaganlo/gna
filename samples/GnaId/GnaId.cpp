//*****************************************************************************
//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation
//
// The source code contained or described herein and all documents related
// to the source code ("Material") are owned by Intel Corporation or its suppliers
// or licensors. Title to the Material remains with Intel Corporation or its suppliers
// and licensors. The Material contains trade secrets and proprietary
// and confidential information of Intel or its suppliers and licensors.
// The Material is protected by worldwide copyright and trade secret laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed in any way
// without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise. Any license under such intellectual property rights must
// be express and approved by Intel in writing.
//*****************************************************************************

#include "gna-api.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <algorithm>
#include <string>

void print_outputs(
    int32_t *outputs,
    uint32_t nRows,
    uint32_t nColumns
)
{
    printf("\nOutputs:\n");
    for(uint32_t i = 0; i < nRows; ++i)
    {
        for(uint32_t j = 0; j < nColumns; ++j)
        {
            printf("%d\t", outputs[i*nColumns + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

int wmain(int argc, wchar_t *argv[])
{
    gna_status_t status = GNA_SUCCESS;

    // open the device
    gna_device_id gna_handle;

    return 0;
}

#define EVALUATOR(x)  #x
#define STRINGIFY(NAME)    EVALUATOR(NAME)

static std::map<gna_api_version, std::string> ApiVersionNames =
{
    {GNA_API_1_0, "GNA API 1.0"},
    {GNA_API_2_0, "GNA API 2.0"},
    //{GNA_API_2_1_S, "GNA API 2.1 Server"},
    {GNA_API_3_0, "GNA API 3.0"},

    {GNA_API_VERSION_COUNT, "Unknown GNA API version"},
};

typedef struct _api_properties_info
{
    uint32_t value;
    gna_property_type type;
    std::string propertyName;
} api_properties_info;


// defined by build variables
#define GNA_API_VERSION_CURRENT (GNA_API_3_0)
#define GNA_API_BUILD_CURRENT 20001100
#define GNA_API_THREAD_COUNT_CURRENT 1

const static std::map<gna_api_property, api_properties_info> ApiProperties =
{
    {GNA_API_VERSION,
        {static_cast<uint32_t>(GNA_API_VERSION_CURRENT), GNA_API_VERSION_T, STRINGIFY(GNA_API_VERSION)}},
    {GNA_API_BUILD,
        {GNA_API_BUILD_CURRENT, GNA_UINT32_T, STRINGIFY(GNA_API_BUILD)}},
    {GNA_API_THREAD_COUNT,
        {GNA_API_THREAD_COUNT_CURRENT, GNA_UINT32_T, STRINGIFY(GNA_API_THREAD_COUNT)}},
    {GNA_API_THREAD_COUNT_MAX,
        {8, GNA_UINT32_T, STRINGIFY(GNA_API_THREAD_COUNT_MAX)}},
    {GNA_API_PROPERTY_COUNT,
        {UINT32_MAX, GNA_UINT32_T, "UNKNOWN GNA_API_PROPERTY"}},
};

gna_api_version GetApiVersionFromValue(void * poropertyValue)
{
    auto version = static_cast<gna_api_version*>(poropertyValue);
    return std::max(*version, GNA_API_VERSION_COUNT);
}

gna_status_t GnaGetDeviceCount(uint32_t * deviceCount)
{
    if (nullptr != deviceCount)
    {
        *deviceCount = 2;
        return GNA_SUCCESS;
    }

    return GNA_NULLARGNOTALLOWED;
}

gna_status_t GnaGetApiProperty(gna_api_property property, void * poropertyValue, gna_property_type * propertyValueType)
{
    if (property < ApiProperties.size() && nullptr != poropertyValue && nullptr != propertyValueType)
    {
        auto value = static_cast<uint32_t*>(poropertyValue);
        //*value = ApiProperties[property];
        //*propertyValueType = ApiPropertiesTypes[property];

        return GNA_SUCCESS;
    }

    return  GNA_NULLARGNOTALLOWED;
}

gna_status_t GnaSetApiProperty(gna_api_property property, void * poropertyValue)
{
    if (property < ApiProperties.size() && nullptr != poropertyValue)
    {
        auto value = GetApiVersionFromValue(poropertyValue);
        //ApiProperties[property] = value;

        return GNA_SUCCESS;
    }

    return  GNA_NULLARGNOTALLOWED;
}

gna_status_t GnaApiPropertyNameToString(gna_api_property property, char const ** propertyString)
{
    if (property < ApiProperties.size() && nullptr != propertyString)
    {
      /*  auto value = ApiPropertiesNames[property];
        *propertyString = value.c_str();*/

        return GNA_SUCCESS;
    }

    return  GNA_NULLARGNOTALLOWED;
}

//const char* GetApiVersionString(gna_api_version version)
//{
//    auto versionString = std::string();
//    GnaIsFlagSet()
//}

gna_status_t GnaApiPropertyValueToString(gna_api_property property, void * poropertyValue, char const ** propertyString)
{
    if (property < ApiProperties.size() && nullptr != poropertyValue && nullptr != propertyString)
    {
        if (GNA_API_VERSION == property)
        {
            auto value = GetApiVersionFromValue(poropertyValue);
            *propertyString = ApiVersionNames[value].c_str();
        }
        //else
        //{
        //    auto value = static_cast<uint32_t*>(poropertyValue);
        //    //sprintf_s()
        //    *propertyString =  *value;
        //}


        return GNA_SUCCESS;
    }

    return  GNA_NULLARGNOTALLOWED;
}

