/*
 @copyright

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-capability-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_CAPABILITY_API Capability API

 API for querying capabilities of hardware devices and library.

 @{
 *****************************************************************************/

#ifndef __GNA2_CAPABILITY_API_H
#define __GNA2_CAPABILITY_API_H

#include "gna2-common-api.h"

#include <stdint.h>

enum GnaDeviceGeneration;

/**
 List of device generations.

 Generation determines the set of capabilities common amongst multiple device versions.
 @see GnaDeviceVersion.
 */
enum GnaDeviceGeneration
{
    /**
     Legacy device supporting only Gaussian Mixture Models scoring.
     */
    GnaDeviceGenerationGmm = 0x010,

    /**
     Initial GNA device generation with no CNN support.
     Backward compatible with ::GnaDeviceGenerationGmm.
     */
    GnaDeviceGeneration0x9 = 0x090,

    /**
     First fully featured GNA device generation.
     Backward compatible with ::GnaDeviceGeneration0x9.
     */
    GnaDeviceGeneration1x0 = 0x100,

    /**
     Embedded device with same feature set as ::GnaDeviceGenerationEmbedded1x0.
     */
    GnaDeviceGenerationEmbedded1x0 = 0x10E,

    /**
     Fully featured second GNA device generation.
     Backward compatible with ::GnaDeviceGenerationEmbedded1x0.
     */
    GnaDeviceGeneration2x0 = 0x200,

    /**
     Embedded device with same feature set as ::GnaDeviceGeneration2x0.
     */
    GnaDeviceGenerationEmbedded2x1 = 0x21E,

    /**
     Fully featured third GNA device generation.
     Partially compatible with ::GnaDeviceGeneration2x0.
     */
    GnaDeviceGeneration3x0 = 0x300,

    /**
     Devices embedded on PCH/ACE, with same feature set as ::GnaDeviceGeneration3x0.
     */
    GnaDeviceGenerationEmbedded3x0 = 0x30E,

    /**
     Devices embedded on PCH/ACE, with same feature set as ::GnaDeviceGeneration3x0
     and autonomous extension.
     */
    GnaDeviceGenerationAutonomus3x1 = 0x31F,
};

/**
 Generation of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @see
 GNA_DEFAULT_DEVICE_VERSION.

 @note
 Usually it will be the latest existing GNA generation (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA_DEFAULT_DEVICE_GENERATION GnaDeviceGeneration3x0

#endif // __GNA2_CAPABILITY_API_H

/**
 @}
 @}
 */

// / * *
// List of API versions.
// */
//enum GnaApiVersion
//{
//    / **
//     Previous GNA API 1.0.
//     */
//    GNA_API_1_0 = 1,
//
//    /* *
//     Current GNA API 2.0.
//     */
//    GNA_API_2_0 = 2,
//
//    /* *
//     Indicates that API version is not supported.
//     */
//    GNA_API_VERSION_NOT_SUPPORTED = GNA_NOT_SUPPORTED,
//};
//
//// Binary flags
//typedef enum _memory_mode
//{
//    GNA_MEMORY_NOT_SUPPORTED = GNA_NOT_SUPPORTED,
//    GNA_MEMORY_MAPPED = 1,
//    GNA_MEMORY_DIRECT = 2,
//    GNA_MEMORY_DEDICATED = 4, // Server device built-in memory
//    GNA_MEMORY_FANCY_MODE = 8, // ACE extension?
//} gna_memory_mode;

//typedef enum _api_property
//{
//    GNA_API_VERSION,                            // GNA_API_VERSION_T
//    GNA_API_BUILD,                              // GNA_UINT32_T
//    GNA_API_THREAD_COUNT,                       // GNA_UINT32_T
//    GNA_API_THREAD_COUNT_MAX,                   // GNA_UINT32_T
//
//    GNA_API_PROPERTY_COUNT
//} gna_api_property;

//enum GnaDevicePropertyType
//{
//    GNA_DEVICE_AVAILABLE_COUNT,                 // GNA_UINT32_T
//    GNA_DEVICE_ACTIVE_COUNT,                    // GNA_UINT32_T
//    GNA_DEVICE_PROFILE,                         // GNA_DEVICE_PROFILE_T
//    GNA_DEVICE_VERSION,                         // GNA_DEVICE_GENERATION_T
//    GNA_DEVICE_DRIVER_BUILD,                    // GNA_UINT32_T
//    GNA_DEVICE_CLOCK_FREQUENCY,                 // GNA_UINT32_T
//    GNA_DEVICE_COMPUTE_ENGINE_COUNT,            // GNA_UINT32_T
//    GNA_DEVICE_ACTIVATION_ENGINE_COUNT,         // GNA_UINT32_T
//    GNA_DEVICE_POOLING_ENGINE_COUNT,            // GNA_UINT32_T
//    GNA_DEVICE_STREAM_COUNT,                    // GNA_UINT32_T
//    GNA_DEVICE_INPUT_BUFFER_SIZE,               // GNA_UINT64_T
//    GNA_DEVICE_MEMORY_MODE,                     // GNA_MEMORY_MODE_T
//    GNA_DEVICE_MEMORY_DEDICATED_SIZE,           // GNA_UINT64_T
//    GNA_DEVICE_MEMORY_REGIONS_COUNT,            // GNA_UINT32_T
//    GNA_DEVICE_MEMORY_SUPPORTED_SIZE,            // GNA_UINT32_T
//    GNA_DEVICE_MODEL_COUNT_MAX,                 // GNA_UINT64_T
//    // ANNA
//    GNA_DEVICE_EXT_,           // GNA_UINT32_T
//};

///** Maximum number of requests that can be enqueued before retrieval */
//const uint32_t GNA_REQUEST_QUEUE_LENGTH = 64;
//
///** Maximum supported time of waiting for a request in milliseconds. */
//const uint32_t GNA_REQUEST_TIMEOUT_MAX = 180000;
//
//enum GnaPropertyType
//{
//    /**
//     Determines if property is supported in given context.
//
//     A single char value, where 0 stands for False and 1 for True.
//     */
//    GNA_PROPERTY_IS_SUPORTED = 0,
//
//    /**
//     Current value of the property
//
//     A single int64_t value.
//     */
//    GNA_PROPERTY_CURRENT_VALUE = 1,
//
//    /**
//     Default value of a parameter, when not set by the user.
//
//     A single int64_t value.
//     */
//    GNA_PROPERTY_DEFAULT_VALUE = 2,
//
//    /**
//     Minimal valid value (inclusive).
//
//     A single int64_t value.
//     */
//    GNA_PROPERTY_MINIMUM = 3,
//
//    /**
//     Maximal valid value (inclusive).
//
//     A single int64_t value.
//     */
//    GNA_PROPERTY_MAXIMUM = 4,
//
//    /**
//     Multiplicity (or step) of valid values.
//
//     A single int64_t value.
//     */
//    GNA_PROPERTY_MULTIPLICITY = 5,
//
//    /**
//     Required alignment of data buffer pointers in bytes.
//
//     A single int64_t value.
//    */
//    GNA_PROPERTY_ALIGNMENT = GNA_PROPERTY_MULTIPLICITY,
//
//    /**
//     Set (array) of valid values, applicable mostly for enumerations.
//     @see GNA_PROPERTY_VALUE_SET_SIZE.
//
//     An array of GNA_PROPERTY_VALUE_SET_SIZE elements, each single uint64_t value.
//     */
//    GNA_PROPERTY_VALUE_SET = 6,
//
//    /**
//     The size of the valid values set, in terms of elements.
//     @see GNA_PROPERTY_VALUE_SET.
//
//     A single uint64_t value.
//     */
//    GNA_PROPERTY_VALUE_SET_SIZE = 7,
//
//    /**
//     Special type, used where property is not applicable or unnecessary.
//     */
//    GNA_PROPERTY_NONE = GNA_DISABLED,
//};
//
///**
// Determines the parameters of GNA Properties in TLV-like format.
// */
//struct GnaProperty
//{
//    enum GnaPropertyType Type;
//
//    uint32_t Size;
//
//    void * Value;
//};


//**
// * Test if given mode is set amongst flags
// *
// * @modeFlags   A value or bitwise OR of more values from GNA mode enumeration.
// * @mode        A tested mode value from GNA mode enumeration.
// * @return true if mode is set, false otherwise
//*/
//inline bool GnaIsFlagSet(uint32_t modeFlags, uint32_t mode)
//{
//    if (modeFlags & mode || GNA_NOT_SUPPORTED == mode)
//    {
//        return true;
//    }
//    return false;
//}
//
// TODO:enable querying some properties on non-available devices like SueScreek
//
//
// dedicated query functions
//GNA_API enum GnaStatus GnaGetApiProperty(
//    gna_api_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//// optional
//GNA_API enum GnaStatus GnaSetApiProperty(
//    gna_api_property property,
//    void* poropertyValue);                      // value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA_LAYER_POOLING_MODE"
//GNA_API enum GnaStatus GnaApiPropertyNameToString(
//    gna_api_property property,
//    char const ** propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA_POOLING_MAX | GNA_POOLING_SUM"
//GNA_API enum GnaStatus GnaApiPropertyValueToString(
//    gna_api_property property,
//    void* poropertyValue,                       // value of property
//    char const ** propertyString);               // [out] c-string containing property value, allocated by GNA

//GNA_API enum GnaStatus GnaGetDeviceProperty(
//    uint32_t device,                       // id/index of device <0;GNA_DEVICE_AVAILABLE_COUNT-1>
//    enum GnaDevicePropertyType capability,
//    enum GnaPropertyType property,
//    struct GnaProperty * deviceProperty);
//
//GNA_API enum GnaStatus GnaGetDeviceProperty(
//    uint32_t device,                       // id/index of device <0;GNA_DEVICE_AVAILABLE_COUNT-1>
//    enum GnaDevicePropertyType capability,
//    struct GnaProperty * deviceProperties);
//
//GNA_API enum GnaStatus GnaSetDeviceProperty(
//    uint32_t device,
//    gna_device_property property,
//    void* poropertyValue);                      // value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA_LAYER_POOLING_MODE"
//GNA_API enum GnaStatus GnaDevicePropertyNameToString(
//    gna_device_property property,
//    char const * propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA_POOLING_MAX | GNA_POOLING_SUM"
//GNA_API enum GnaStatus GnaDevicePropertyValueToString(
//    gna_device_property property,
//    void* poropertyValue,                       // value of property
//    char const * propertyString);               // [out] c-string containing property value, allocated by GNA
//
//GNA_API enum GnaStatus GnaGetLayerProperty(
//    uint32_t device,
//    GnaOperationMode layerOperation,
//    gna_layer_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNA_API enum GnaStatus GnaSetLayerProperty(
//    uint32_t device,
//    GnaOperationMode layerOperation,
//    gna_layer_property property,
//    void* poropertyValue);                      // value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA_LAYER_POOLING_MODE"
//GNA_API enum GnaStatus GnaLayerPropertyNameToString(
//    gna_layer_property property,
//    char const * propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA_POOLING_MAX | GNA_POOLING_SUM"
//GNA_API enum GnaStatus GnaLayerPropertyValueToString(
//    gna_layer_property property,
//    void* poropertyValue,                       // value of property
//    char const * propertyString);               // [out] c-string containing property value, allocated by GNA
//
//
//// Query hardware device properties even if not present in system, like SueCreek
//GNA_API enum GnaStatus GnaGetHardwareDeviceProperty(
//    enum GnaDeviceGeneration generation,         // hardware device generation identifier, for not present devices
//    gna_device_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNA_API enum GnaStatus GnaGetHardwareLayerProperty(
//    enum GnaDeviceGeneration generation,         // hardware device generation identifier, for not present devices
//    GnaOperationMode layerOperation,
//    gna_layer_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
