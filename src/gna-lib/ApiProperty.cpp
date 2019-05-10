////*****************************************************************************
////
//// INTEL CONFIDENTIAL
//// Copyright 2018 Intel Corporation
////
//// The source code contained or described herein and all documents related
//// to the source code ("Material") are owned by Intel Corporation or its suppliers
//// or licensors. Title to the Material remains with Intel Corporation or its suppliers
//// and licensors. The Material contains trade secrets and proprietary
//// and confidential information of Intel or its suppliers and licensors.
//// The Material is protected by worldwide copyright and trade secret laws and treaty
//// provisions. No part of the Material may be used, copied, reproduced, modified,
//// published, uploaded, posted, transmitted, distributed, or disclosed in any way
//// without Intel's prior express written permission.
////
//// No license under any patent, copyright, trade secret or other intellectual
//// property right is granted to or conferred upon you by disclosure or delivery
//// of the Materials, either expressly, by implication, inducement, estoppel
//// or otherwise. Any license under such intellectual property rights must
//// be express and approved by Intel in writing.
////*****************************************************************************
//
//#include <algorithm>
//
//#include "ApiProperty.hpp"
//
//using namespace GNA;
//
//PopertyType::PopertyType(gna_property_type id, std::string name, size_t size, const char* const printFormat) :
//    Id{ id },
//    Name{ name },
//    Size{ size },
//    PrintFormat{ printFormat }
//{
//}
//
//const std::map<gna_property_type, PopertyType> PopertyType::Types =
//{
//    {GNA_TYPE_NOT_SUPPORTED, {GNA_TYPE_NOT_SUPPORTED, "GNA_TYPE_NOT_SUPPORTED", 0ui64, "%s"}},
//    {GNA_UINT8_T, {GNA_UINT8_T, "uint8_t", sizeof(uint8_t), "%uhh"}},
//    {GNA_UINT16_T, {GNA_UINT16_T, "uint16_t", sizeof(uint16_t), "%uh"}},
//    {GNA_UINT32_T, {GNA_UINT32_T, "uint32_t", sizeof(uint32_t), "%u"}},
//    {GNA_UINT64_T, {GNA_UINT64_T, "uint64_t",sizeof(uint64_t), "%llu"}},
//    {GNA_BOOL_T, {GNA_BOOL_T, "bool", sizeof(bool), "%s"}},
//    {GNA_API_VERSION_T, {GNA_API_VERSION_T, "gna_api_version", sizeof(gna_api_version), "%s"}},
//
//    {GNA_DEVICE_GENERATION_T, {GNA_DEVICE_GENERATION_T, "gna_device_generation", sizeof(gna_device_generation), "%s"}},
//    {GNA_MEMORY_MODE_T, {GNA_MEMORY_MODE_T, "gna_memory_mode", sizeof(gna_memory_mode), "%s"}},
//    {GNA_DATA_MODE_T, {GNA_DATA_MODE_T, "gna_data_mode", sizeof(gna_data_mode), "%s"}},
//    {GNA_BIAS_MODE_T, {GNA_BIAS_MODE_T, "gna_bias_mode", sizeof(gna_bias_mode), "%s"}},
//    {GNA_POOLING_MODE_T, {GNA_POOLING_MODE_T, "gna_pooling_mode", sizeof(gna_pooling_mode), "%s"}},
//    {GNA_TENSOR_ORDER_T, {GNA_TENSOR_ORDER_T, "gna_tensor_order", sizeof(gna_tensor_order), "%s"}},
//};
//
//
//Property::Property(/*PropertyDomain& domain, */const std::string name, const PopertyType& type, void* valueIn) :
//    Name{ name },
//    Type{ type }
//    //Domain{ domain }
//{
//    if (nullptr != valueIn)
//    {
//        throw GnaException(GNA_NULLARGNOTALLOWED);
//    }
//    value = calloc(1, Type.Size);
//    if (nullptr == value)
//    {
//        throw GnaException(Gna2StatusResourceAllocationError);
//    }
//    auto err = memcpy_s(value, Type.Size, valueIn, Type.Size);
//    if (0 != err)
//    {
//        throw GnaException(Gna2StatusResourceAllocationError);
//    }
//}
//
//inline Property::~Property()
//{
//    if (nullptr != value)
//    {
//        free(value);
//    }
//}
//
//// defined by build variables
//#define GNA_API_BUILD_CURRENT 20001100
//#define GNA_API_THREAD_COUNT_CURRENT 1
//
//static const EnumerableProperty ApiVersion =
//{
//    std::string{STRINGIFY(GNA_API_VERSION)},
//    PopertyType::Types.at(GNA_API_VERSION_T),
//    GNA_API_3_0,
//    {
//        {GNA_API_1_0, "GNA API 1.0"},
//        {GNA_API_2_0, "GNA API 2.0"},
//        //{GNA_API_2_1_S, "GNA API 2.1 Server"},
//        {GNA_API_3_0, "GNA API 3.0"},
//        {GNA_API_VERSION_COUNT, "Unknown GNA API version"},
//    }
//};
//
//const std::map<PropertyDomainKind, std::string> PropertyDomain::PropertyDomainNames =
//{
//    {GNA_API_PROPERTY, "API property"},
//    {GNA_DEVICE_PROPERTY, "Hardware Device property"},
//    {GNA_LAYER_PROPERTY, "Hardware Layer property"}
//};
//
//PropertyDomain::PropertyDomain(PropertyDomainKind kind, const std::map<uint32_t, PropertyPtr> properties) :
//    Kind{ kind },
//    Name{ PropertyDomainNames.at(kind) },
//    Properties{ properties }
//{
//
//}
//
////static const PropertyDomain ApiPropertyDomain =
////{
////    GNA_API_PROPERTY,
////    {
////        {
////            GNA_API_VERSION,
////            std::make_unique<EnumerableProperty>(ApiVersion)
////        },
////    /*{GNA_API_BUILD, STRINGIFY(GNA_API_BUILD)},
////    {GNA_API_THREAD_COUNT, STRINGIFY(GNA_API_THREAD_COUNT)},
////    {GNA_API_THREAD_COUNT_MAX, STRINGIFY(GNA_API_THREAD_COUNT_MAX)},
////    {GNA_API_PROPERTY_COUNT, "UNKNOWN GNA_API_PROPERTY"},*/
////    }
////};
//
//    //{
//    //    {
//    //        GNA_API_PROPERTY,
//    //        {
//
//    //    },
//    //    {GNA_DEVICE_PROPERTY, {}},
//    //    {GNA_LAYER_PROPERTY, {}},
//    //};
//
////{
////    {GNA_API_VERSION, STRINGIFY(GNA_API_VERSION)}},
////    {GNA_API_BUILD, STRINGIFY(GNA_API_BUILD)}},
////    {GNA_API_THREAD_COUNT, STRINGIFY(GNA_API_THREAD_COUNT)}},
////    {GNA_API_THREAD_COUNT_MAX, STRINGIFY(GNA_API_THREAD_COUNT_MAX)}},
////    {GNA_API_PROPERTY_COUNT, "UNKNOWN GNA_API_PROPERTY"}},
////};
//
//
//
//const std::map<PropertyDomainKind, PropertyDomain> Capabilities =
//{
//
//};
//
//
////
////ApiProperty::ApiProperty()
////{
////}
////
////ApiProperty::~ApiProperty()
////{
////}
////
//////std::map<gna_api_version, std::string> ApiProperty::ApiVersionNames =
//////{
//////    {GNA_API_1_0, "GNA API 1.0"},
//////    {GNA_API_2_0, "GNA API 2.0"},
////////    {GNA_API_2_1_S, "GNA API 2.1 Server"},
//////    {GNA_API_3_0, "GNA API 3.0"},
//////    {GNA_API_VERSION_COUNT, "Unknown GNA API version"},
//////};
////
////// defined by build variables
////#define GNA_API_VERSION_CURRENT (GNA_API_3_0)
////#define GNA_API_BUILD_CURRENT 20001100
////#define GNA_API_THREAD_COUNT_CURRENT 1
////
////std::map<gna_api_property, ApiProperty::ApiInfo> ApiProperty::ApiProperties =
////{
////    {GNA_API_VERSION,
////        {static_cast<uint32_t>(GNA_API_VERSION_CURRENT), GNA_API_VERSION_T, STRINGIFY(GNA_API_VERSION)}},
////    {GNA_API_BUILD,
////        {GNA_API_BUILD_CURRENT, GNA_UINT32_T, STRINGIFY(GNA_API_BUILD)}},
////    {GNA_API_THREAD_COUNT,
////        {GNA_API_THREAD_COUNT_CURRENT, GNA_UINT32_T, STRINGIFY(GNA_API_THREAD_COUNT)}},
////    {GNA_API_THREAD_COUNT_MAX,
////        {8, GNA_UINT32_T, STRINGIFY(GNA_API_THREAD_COUNT_MAX)}},
////    {GNA_API_PROPERTY_COUNT,
////        {UINT32_MAX, GNA_UINT32_T, "UNKNOWN GNA_API_PROPERTY"}},
////};
////
////status_t ApiProperty::GnaGetApiProperty(gna_api_property property, void * poropertyValue, gna_property_type * propertyValueType)
////{
////    if (static_cast<size_t>(property) < ApiProperties.size() && nullptr != poropertyValue && nullptr != propertyValueType)
////    {
////        auto value = static_cast<uint32_t*>(poropertyValue);
////        *value = ApiProperties[property].value;
////        *propertyValueType = ApiProperties[property].type;
////
////        return GNA_SUCCESS;
////    }
////
////    return  GNA_NULLARGNOTALLOWED;
////}
////
////
////
//////GNAAPI status_t GnaGetApiProperty(gna_api_property property, void * poropertyValue, gna_property_type * propertyValueType)
//////{
//////    if (property < ApiProperties.size() && nullptr != poropertyValue && nullptr != propertyValueType)
//////    {
//////        auto value = static_cast<uint32_t*>(poropertyValue);
//////        *value = ApiProperties[property];
//////        *propertyValueType = ApiPropertiesTypes[property];
//////
//////        return GNA_SUCCESS;
//////    }
//////
//////    return  GNA_NULLARGNOTALLOWED;
//////}
//////
//////GNAAPI status_t GnaSetApiProperty(gna_api_property property, void * poropertyValue)
//////{
//////    if (property < ApiProperties.size() && nullptr != poropertyValue)
//////    {
//////        auto value = GetApiVersionFromValue(poropertyValue);
//////        ApiProperties[property] = value;
//////
//////        return GNA_SUCCESS;
//////    }
//////
//////    return  GNA_NULLARGNOTALLOWED;
//////}
//////
//////GNAAPI status_t GnaApiPropertyNameToString(gna_api_property property, char const ** propertyString)
//////{
//////    if (property < ApiProperties.size() && nullptr != propertyString)
//////    {
//////        auto value = ApiPropertiesNames[property];
//////        *propertyString = value.c_str();
//////
//////        return GNA_SUCCESS;
//////    }
//////
//////    return  GNA_NULLARGNOTALLOWED;
//////}
//////
////////const char* GetApiVersionString(gna_api_version version)
////////{
////////    auto versionString = std::string();
////////    GnaIsFlagSet()
////////}
//////
//////GNAAPI status_t GnaApiPropertyValueToString(gna_api_property property, void * poropertyValue, char const ** propertyString)
//////{
//////    if (property < ApiProperties.size() && nullptr != poropertyValue && nullptr != propertyString)
//////    {
//////        if (GNA_API_VERSION == property)
//////        {
//////            auto value = GetApiVersionFromValue(poropertyValue);
//////            *propertyString = ApiVersionNames[value].c_str();
//////        }
//////        else
//////        {
//////            auto value = static_cast<uint32_t*>(poropertyValue);
//////            sprintf_s()
//////            *propertyString =  *value;
//////        }
//////
//////
//////        return GNA_SUCCESS;
//////    }
//////
//////    return  GNA_NULLARGNOTALLOWED;
//////}
////
