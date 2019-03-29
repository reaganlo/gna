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
 @file gna2-model-export-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_MODEL_API
 @{
 *****************************************************************************

 @addtogroup GNA2_MODEL_EXPORT_API Model Export API

 API for exporting GNA model for embedded devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_MODEL_EXPORT_API_H
#define __GNA2_MODEL_EXPORT_API_H

#include "gna2-common-api.h"

#if !defined(_WIN32)
#include <assert.h>
#endif
#include <stdint.h>

/**
 Creates configuration for model exporting.

 Export configuration allows to configure all the parameters necessary
 to export components of one or more models.
 Use GnaModelExportConfigSet*() functions to configure parameters. Parameters
 can be modified/overridden for existing configuration to export model
 with modified properties.

 @warning
    User is responsible for releasing allocated memory buffers.

 @param userAllocator User provided memory allocator.
 @param [out] exportConfigId Identifier of created export configuration.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelExportConfigCreate(
    GnaUserAllocator userAllocator,
    uint32_t * exportConfigId);

/**
 Releases export configuration and all its resources.

 @param exportConfigId Identifier of export configuration to release.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelExportConfigRelease(
    uint32_t exportConfigId);

/**
 Sets source model(s) to export.

 - Model will be validated against provided device.
 - Model(s) should be created through standard API GnaModelCreate() function.

 @param exportConfigId Identifier of export configuration to set.
 @param sourceDeviceIndex Id of the device on which the exported model was created.
    Use GNA_DISABLED to export model from all available devices at one.
 @param sourceModelId Id of the source model, created previously with GnaModelCreate() function.
     Use GNA_DISABLED to export all models from given device at one.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId);

/**
 Sets version of the device that exported model will be used with.

 - Model will be validated against provided target device.

 @param exportConfigId Identifier of export configuration to set.
 @param targetDeviceVersion Device on which model will be used.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum GnaDeviceVersion targetDeviceVersion);

enum GnaModelExportComponent;

/**
 Exports the model(s) component.

 All exported model components are saved into memory allocated on user side by userAllocator.

 @warning
    User is responsible for releasing allocated memory buffers (exportBuffer).

 @param exportConfigId Identifier of export configuration used.
 @param componentType What component should be exported.
 @param [out] exportBuffer Memory allocated by userAllocator with exported layer descriptors.
 @param [out] exportBufferSize The size of exportBuffer in bytes.
 @return Status of the operation.
 */
GNA_API enum GnaStatus GnaModelExport(
    uint32_t exportConfigId,
    enum GnaModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize);

/**
 Determines the type of the component to export.
 */
enum GnaModelExportComponent
{
    /**
     Hardware layer descriptors will be exported.
     */
    GnaModelExportComponentLayerDescriptors = GNA_DEFAULT,

    /**
     Header describing layer descriptors will be exported.
     */
    GnaModelExportComponentLayerDescriptorHeader = 1,

    /**
     Hardware layer descriptors in legacy SueCreek format will be exported.
     */
    GnaModelExportComponentLegacySueCreekLayerDescriptors = 2,

    /**
     Header describing layer descriptors in legacy SueCreek format will be exported.
     */
    GnaModelExportComponentLegacySueCreekLayerDescriptorHeader = 3,
};

//
/// * *
// Exports the hardware-consumable model in layer descriptor format.
//
// Model should be created through standard API GnaModelCreate() function
//
// @param exportConfigId Identifier of export configuration used.
// @param sourceDeviceIndex Id of the device on which the exported model was created.
// @param sourceModelId Id of the source model, created previously with GnaModelCreate() function.
// @param [out] layerDescriptorBuffer Memory allocated by userAllocator with exported layer descriptors.
// @param [out] layerDescriptorBufferSize The size of layerDescriptorBuffer in bytes.
// @return Status of the operation.
// */
//GNA_API enum GnaStatus GnaModelExportLayerDescriptors(
//    uint32_t exportConfigId,
//    uint32_t sourceDeviceIndex,
//    uint32_t sourceModelId,
//    void * layerDescriptorBuffer,
//    uint32_t * layerDescriptorBufferSize);
//
/// * *
// Exports the layer descriptor header.
//
// Format of the model header is dependent of export configuration.
//
// @param exportConfigId Identifier of export configuration used.
// @param sourceDeviceIndex Id of the device on which the exported model was created.
// @param sourceModelId Id of the source model, created previously with GnaModelCreate() function.
// @param [out] layerDescriptorHeaderBuffer Memory allocated by userAllocator with exported layer descriptor header.
// @param [out] layerDescriptorHeaderBufferSize The size of layerDescriptorHeaderBuffer in bytes.
// @return Status of the operation.
// */
//GNA_API enum GnaStatus GnaModelExportLayerDescriptorHeader(
//    uint32_t exportConfigId,
//    uint32_t sourceDeviceIndex,
//    uint32_t sourceModelId,
//    void * layerDescriptorHeaderBuffer,
//    uint32_t * layerDescriptorHeaderBufferSize);

#endif // __GNA2_MODEL_EXPORT_API_H

/**
 @}
 @}
 @}
 */
