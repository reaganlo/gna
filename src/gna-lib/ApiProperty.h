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
//#include <cstring>
//#include <cstdlib>
//#include <cstdio>
//#include <map>
//#include <string>
//#include <memory>
//
//
//#include "GnaException.h"
//
//namespace GNA
//{
//
//#define EVALUATOR(x)  #x
//#define STRINGIFY(NAME)    EVALUATOR(NAME)
//
//class PropertyDomain;
//
//class PopertyType
//{
//public:
//    static const std::map<gna_property_type, PopertyType> Types;
//
//    const gna_property_type Id;
//    const std::string Name;
//    const size_t Size;
//    const char* const PrintFormat;
//
//protected:
//    PopertyType(gna_property_type id, std::string name, size_t size, const char* const printFormat);
//};
//
//// Domain type e.g.  gna_api_property, gna_device_property, gna_layer_property
//// Kind e.g. GNA_API_VERSION, GNA_DEVICE_DRIVER_BUILD or GNA_LAYER_OUTPUT_TENSOR_DIM_W_MAX
//// Type of property Kind e.g. uint32_t, gna_api_version
//class Property
//{
//public:
//    Property(/*PropertyDomain& domain,*/ const std::string name, const PopertyType& type, void* valueIn);
//    ~Property();;
//
//    const std::string Name;
//    const PopertyType& Type;
//    //PropertyDomain& Domain;
//    virtual const std::string ToString() const
//    {
//        const size_t size = 256;
//        char buffer[size];
//        sprintf_s(buffer, size, Type.PrintFormat, *(static_cast<uint8_t*>(value)));
//        auto valueString = std::string(buffer);
//        return valueString;
//    }
//
//    template <typename T> T Get() const
//    {
//        return static_cast<T>(*value);
//    }
//
//    template <typename T> void Set(T newValue)
//    {
//        *value = newValue;
//    }
//
//protected:
//    void* value;
//};
//
//class EnumerableProperty : protected Property
//{
//public:
//    EnumerableProperty(const std::string name, const PopertyType& type, uint32_t valueIn, std::map<uint32_t, std::string> valuesNames) :
//        Property{ name, type, static_cast<void*>(&valueIn) },
//        ValuesNames{ valuesNames }
//    {
//
//    }
//    const std::string ToString() const override
//    {
//        const size_t size = 256;
//        char buffer[size];
//        sprintf_s(buffer, size, Type.PrintFormat, ValuesNames.at(*(static_cast<gna_api_version*>(value))).c_str());
//        auto valueString = std::string(buffer);
//        return valueString;
//    }
//protected:
//    const std::map<uint32_t, std::string> ValuesNames;
//};
//
//class MultiEnumerableProperty : EnumerableProperty
//{
//public:
//    MultiEnumerableProperty(const std::string name, const PopertyType& type, uint32_t valueIn, std::map<uint32_t, std::string> valuesNames) :
//        EnumerableProperty{ name, type, valueIn, valuesNames }
//    {
//    }
//    const std::string ToString() const override
//    {
//        const size_t size = 256;
//        char buffer[size];
//
//        // TODO: handle multiple bit set
//        sprintf_s(buffer, size, Type.PrintFormat, ValuesNames.at(*(static_cast<gna_api_version*>(value))).c_str());
//        auto valueString = std::string(buffer);
//        return valueString;
//    }
//private:
//};
//
//typedef enum _PropertyDomainKind
//{
//    GNA_API_PROPERTY,
//    GNA_DEVICE_PROPERTY,
//    GNA_LAYER_PROPERTY,
//} PropertyDomainKind;
//
////typedef std::unique_ptr<Property> PropertyPtr;
//typedef Property* PropertyPtr;
//
//class PropertyDomain
//{
//public:
//    PropertyDomain(PropertyDomainKind kind, const std::map<uint32_t, PropertyPtr> properties);
//    ~PropertyDomain() = default;
//
//    static const std::map<PropertyDomainKind, std::string> PropertyDomainNames;
//
//    const PropertyDomainKind Kind;
//    const std::string Name;
//    const std::map<uint32_t, PropertyPtr> Properties;
//};
//
//static const std::map<PropertyDomainKind, PropertyDomain> Capabilities;
//
//}
