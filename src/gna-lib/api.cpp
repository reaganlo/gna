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

#define _COMPONENT_ "GnaApi::"

#include <thread>

#include "Device.h"
#include "Validator.h"

using std::thread;

using namespace GNA;

/******************************************************************************
 *
 * API objects
 *
 *****************************************************************************/

 // TODO: use unique pointer instead
Device Gna2Ultimate;

const char *GNAStatusName[] =
{
    "GNA_SUCCESS - Success: Operation successful, no errors or warnings",
    "GNA_DEVICEBUSY - Warning: Device busy - accelerator is still running, can not enqueue more requests",
    "GNA_SSATURATE - Warning: Scoring saturation - an arithmetic operation has resulted in saturation",
    "GNA_UNKNOWN_ERROR - Error: Unknown error occurred",
    "GNA_ERR_QUEUE - Error: Queue can not create or enqueue more requests",
    "GNA_READFAULT - Error: Scoring data: invalid input",
    "GNA_WRITEFAULT - Error: Scoring data: invalid output buffer",
    "GNA_BADFEATWIDTH - Error: Feature vector: width not supported",
    "GNA_BADFEATLENGTH - Error: Feature vector: length not supported",
    "GNA_BADFEATOFFSET - Error: Feature vector: offset not supported",
    "GNA_BADFEATALIGN - Error: Feature vector: invalid memory alignment",
    "GNA_BADFEATNUM - Error: Feature vector: Number of feature vectors not supported",

    "GNA_INVALIDINDICES - Error: Scoring data: number of active indices  not supported",
    "GNA_DEVNOTFOUND - Error: Device: device not available",
    "GNA_OPENFAILURE - Error: Device: internal error occurred during opening device",
    "GNA_INVALIDHANDLE - Error: Device: invalid handle",
    "GNA_CPUTYPENOTSUPPORTED - Error: Device: processor type not supported",
    "GNA_PARAMETEROUTOFRANGE - Error: Device: GMM Parameter out of Range error occurred",
    "GNA_VAOUTOFRANGE - Error: Device: Virtual Address out of range on DMA ch.",
    "GNA_UNEXPCOMPL - Error: Device: Unexpected completion during PCIe operation",
    "GNA_DMAREQERR - Error: Device: DMA error during PCIe operation",
    "GNA_MMUREQERR - Error: Device: MMU error during PCIe operation",
    "GNA_BREAKPOINTPAUSE - Error: Device: GMM accelerator paused on breakpoint",
    "GNA_BADMEMALIGN - Error: Device: invalid memory alignment",
    "GNA_INVALIDMEMSIZE - Error: Device: requested memory size not supported",
    "GNA_MODELSIZEEXCEEDED - Error: Device: request's model configuration exceeded supported GNA_HW mode limits",
    "GNA_BADREQID - Error: Device: invalid scoring request identifier",
    "GNA_WAITFAULT - Error: Device: wait failed",
    "GNA_IOCTLRESERR - Error: Device: IOCTL result retrieval failed",
    "GNA_IOCTLSENDERR - Error: Device: sending IOCTL failed",
    "GNA_NULLARGNOTALLOWED - Error: NULL argument not allowed",
    "GNA_NULLARGREQUIRED - Error: NULL argument is required",
    "GNA_ERR_MEM_ALLOC1 - Error: Memory: Already allocated, only single allocation per device is allowed",
    "GNA_ERR_RESOURCES - Error: Unable to create new resources",
    "GNA_ERR_NOT_MULTIPLE - Error: Value is not multiple of required value",
    "GNA_ERR_DEV_FAILURE - Error: Critical device error occurred, device has been reset",

    "GMM_BADMEANWIDTH - Error: Mean vector: width not supported",
    "GMM_BADMEANOFFSET - Error: Mean vector: offset not supported",
    "GMM_BADMEANSETOFF - Error: Mean vector: set offset not supported",
    "GMM_BADMEANALIGN - Error: Mean vector: invalid memory alignment",
    "GMM_BADVARWIDTH - Error: Variance vector: width not supported",
    "GMM_BADVAROFFSET - Error: Variance vector: offset not supported",
    "GMM_BADVARSETOFF - Error: Variance vector: set offset not supported",
    "GMM_BADVARSALIGN - Error: Variance vector: invalid memory alignment",
    "GMM_BADGCONSTOFFSET - Error: Gconst: set offset not supported",
    "GMM_BADGCONSTALIGN - Error: Gconst: invalid memory alignment",
    "GMM_BADMIXCNUM - Error: Scoring data: number of mixture components not supported",
    "GMM_BADNUMGMM - Error: Scoring data: number of GMMs not supported",
    "GMM_BADMODE - Error: Scoring data: GMM scoring mode not supported",
    "GMM_CFG_INVALID_LAYOUT - Error: GMM Data layout is invalid",

    "XNN_ERR_NET_LYR_NO - Error: XNN: Not supported number of layers",
    "XNN_ERR_NETWORK_INPUTS - Error: XNN: Network is invalid - input buffers number differs from input layers number",
    "XNN_ERR_NETWORK_OUTPUTS - Error: XNN: Network is invalid - output buffers number differs from output layers number",
    "XNN_ERR_LYR_KIND - Error: XNN: Not supported layer kind",
    "XNN_ERR_LYR_TYPE - Error: XNN: Not supported layer type",
    "XNN_ERR_LYR_CFG - Error: XNN: Invalid layer configuration",
    "XNN_ERR_NO_FEEDBACK - Error: XNN: No RNN feedback buffer specified",
    "XNN_ERR_NO_LAYERS - Error: XNN: At least one layer must be specified",
    "XNN_ERR_GROUPING - Error: XNN: Invalid grouping factor",
    "XNN_ERR_INPUT_BYTES - Error: XNN: Invalid number of bytes per input",
    "XNN_ERR_INT_OUTPUT_BYTES - Error: XNN: Invalid number of bytes per intermediate output",
    "XNN_ERR_OUTPUT_BYTES - Error: XNN: Invalid number of bytes per output",
    "XNN_ERR_WEIGHT_BYTES - Error: XNN: Invalid number of bytes per weight",
    "XNN_ERR_BIAS_BYTES - Error: XNN: Invalid number of bytes per bias",
    "XNN_ERR_BIAS_MULTIPLIER - Error: XNN: Multiplier larger than 255",
    "XNN_ERR_BIAS_INDEX - Error: XNN: Bias Vector index larger than grouping factor",
    "XNN_ERR_PWL_SEGMENTS - Error: XNN: Activation function segment count larger than 128",
    "XNN_ERR_PWL_DATA - Error: XNN: Activation function enabled but segment data not set",
    "XNN_ERR_MM_INVALID_IN - Error: XNN: Invalid input data or configuration in matrix mul. op.",
    "CNN_ERR_FLT_COUNT - Error: CNN Layer: invalid number of filters",
    "CNN_ERR_FLT_STRIDE - Error: CNN Layer: invalid filter stride",
    "CNN_ERR_POOL_STRIDE - Error: CNN Layer: invalid pool stride",

	"UNKNOWN STATUS"          // Status code is invalid"
};

static_assert((NUMGNASTATUS+1) == (sizeof(GNAStatusName)/sizeof(char*)), "Invalid size of GNAStatusName");

/******************************************************************************
 *
 * Library internal objects
 *
 *****************************************************************************/

/******************************************************************************
 *
 * API routines implementation
 *
 *****************************************************************************/

intel_gna_status_t GnaModelCreate(
    gna_device_id       deviceId,
    gna_model*          model,
    gna_model_id*       modelId)
{
    try
    {
        Gna2Ultimate.ValidateSession(deviceId);
        Gna2Ultimate.LoadModel(modelId, model);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaModelRelease(
    gna_model_id        modelId)
{
    try
    {
        Gna2Ultimate.ReleaseModel(modelId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaModelRequestConfigAdd(
    gna_model_id        modelId,
    gna_request_cfg_id* configId)
{
    try
    {
        Gna2Ultimate.CreateConfiguration(modelId, configId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaRequestConfigBufferAdd(
    gna_request_cfg_id  configId,
    gna_buffer_type     type,
    uint32_t            layerIndex,
    void*               address)
{
    try
    {
        Gna2Ultimate.AttachBuffer(configId, type, layerIndex, address);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaRequestConfigActiveListAdd(
    gna_request_cfg_id  configId,
    uint32_t            layerIndex,
    uint32_t            indicesCount,
    uint32_t*           indices)
{
    try
    {
        Gna2Ultimate.AttachActiveList(configId, layerIndex, indicesCount, indices);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaRequestEnqueue(
    gna_request_cfg_id  configId,
    gna_acceleration    accel,
    gna_request_id*     requestId)
{
    try
    {
        acceleration internal_acceleration = static_cast<acceleration>(accel);
        Gna2Ultimate.PropagateRequest(configId, internal_acceleration, requestId);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaRequestWait(
    gna_request_id      requestId,
    gna_timeout         milliseconds)
{
    try
    {
        return Gna2Ultimate.WaitForRequest(requestId, milliseconds);
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

const char* GnaStatusToString(
    intel_gna_status_t status)
{
    return "gna status";
}

void *GnaAlloc(gna_device_id deviceId, uint32_t requestedSize, uint32_t *grantedSize)
{
    try
    {
        //TODO: refactor - to much logic in wrapper
        Gna2Ultimate.ValidateSession(deviceId);
        Expect::NotNull(grantedSize);

        void* buffer = nullptr;
        *grantedSize = Gna2Ultimate.AllocateMemory(requestedSize, &buffer);
        Expect::False(nullptr == buffer || *grantedSize < requestedSize, GNA_ERR_RESOURCES);
        return buffer;
    }
    catch (const GnaException &e)
    {
        ERRS("Memory allocation failed", e.getStatus());
        return nullptr;
    }
    catch (...)
    {
        return nullptr;
    }
}

intel_gna_status_t GnaFree(gna_device_id deviceId)
{
    try
    {
        Gna2Ultimate.ValidateSession(deviceId);
        Gna2Ultimate.FreeMemory();
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

// TODO: instantiate unique pointer
intel_gna_status_t GnaDeviceOpen(
    uint8_t             threadCount,
    gna_device_id*      deviceId)
{
    try
    {
        return Gna2Ultimate.Open(deviceId, threadCount);
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

// TODO: delete unique pointer
intel_gna_status_t GnaDeviceClose(
    gna_device_id deviceId)
{
    try
    {
        Gna2Ultimate.ValidateSession(deviceId);
        Gna2Ultimate.Close();
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}

intel_gna_status_t GnaRequestConfigEnablePerf( gna_request_cfg_id configId, gna_hw_perf_stats perfStat,
    gna_perf_t* perfResults)
{
    try
    {
        Gna2Ultimate.EnableProfiling(configId, perfStat, perfResults);
        return GNA_SUCCESS;
    }
    catch (const GnaException &e)
    {
        return e.getStatus();
    }
    catch (...)
    {
        return GNA_UNKNOWN_ERROR;
    }
}