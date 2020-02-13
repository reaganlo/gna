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
#include "ApiWrapper.h"

#include "Expect.h"

#include "gna2-common-impl.h"

#include <map>


#if !defined(_WIN32)
#include <assert.h>
#endif

/**
 Verifies data sizes used in the API and GNA hardware

 @note If data sizes in an application using API differ from data sizes
       in the API library implementation, scoring will not work properly.
 */
static_assert(1 == sizeof(int8_t), "Invalid size of int8_t");
static_assert(2 == sizeof(int16_t), "Invalid size of int16_t");
static_assert(4 == sizeof(int32_t), "Invalid size of int32_t");
static_assert(1 == sizeof(uint8_t), "Invalid size of uint8_t");
static_assert(2 == sizeof(uint16_t), "Invalid size of uint16_t");
static_assert(4 == sizeof(uint32_t), "Invalid size of uint32_t");

extern const std::map<Gna2Status, std::string> Gna2StatusToStringMap
{
    { Gna2StatusSuccess, "Gna2StatusSuccess" },
    { Gna2StatusWarningDeviceBusy, "Gna2StatusWarningDeviceBusy" },
    { Gna2StatusWarningArithmeticSaturation, "Gna2StatusWarningArithmeticSaturation" },
    { Gna2StatusUnknownError, "Gna2StatusUnknownError" },
    { Gna2StatusNotImplemented, "Gna2StatusNotImplemented" },
    { Gna2StatusIdentifierInvalid, "Gna2StatusIdentifierInvalid" },
    { Gna2StatusNullArgumentNotAllowed, "Gna2StatusNullArgumentNotAllowed" },
    { Gna2StatusNullArgumentRequired, "Gna2StatusNullArgumentRequired" },
    { Gna2StatusResourceAllocationError, "Gna2StatusResourceAllocationError" },
    { Gna2StatusDeviceNotAvailable, "Gna2StatusDeviceNotAvailable" },
    { Gna2StatusDeviceNumberOfThreadsInvalid, "Gna2StatusDeviceNumberOfThreadsInvalid" },
    { Gna2StatusDeviceVersionInvalid, "Gna2StatusDeviceVersionInvalid" },
    { Gna2StatusDeviceQueueError, "Gna2StatusDeviceQueueError" },
    { Gna2StatusDeviceIngoingCommunicationError, "Gna2StatusDeviceIngoingCommunicationError" },
    { Gna2StatusDeviceOutgoingCommunicationError, "Gna2StatusDeviceOutgoingCommunicationError" },
    { Gna2StatusDeviceParameterOutOfRange, "Gna2StatusDeviceParameterOutOfRange" },
    { Gna2StatusDeviceVaOutOfRange, "Gna2StatusDeviceVaOutOfRange" },
    { Gna2StatusDeviceUnexpectedCompletion, "Gna2StatusDeviceUnexpectedCompletion" },
    { Gna2StatusDeviceDmaRequestError, "Gna2StatusDeviceDmaRequestError" },
    { Gna2StatusDeviceMmuRequestError, "Gna2StatusDeviceMmuRequestError" },
    { Gna2StatusDeviceBreakPointHit, "Gna2StatusDeviceBreakPointHit" },
    { Gna2StatusDeviceCriticalFailure, "Gna2StatusDeviceCriticalFailure" },
    { Gna2StatusMemoryAlignmentInvalid, "Gna2StatusMemoryAlignmentInvalid" },
    { Gna2StatusMemorySizeInvalid, "Gna2StatusMemorySizeInvalid" },
    { Gna2StatusMemoryTotalSizeExceeded, "Gna2StatusMemoryTotalSizeExceeded" },
    { Gna2StatusMemoryBufferInvalid, "Gna2StatusMemoryBufferInvalid" },
    { Gna2StatusRequestWaitError, "Gna2StatusRequestWaitError" },
    { Gna2StatusActiveListIndicesInvalid, "Gna2StatusActiveListIndicesInvalid" },
    { Gna2StatusAccelerationModeNotSupported, "Gna2StatusAccelerationModeNotSupported" },
    { Gna2StatusModelConfigurationInvalid, "Gna2StatusModelConfigurationInvalid" },
    { Gna2StatusNotMultipleOf, "Gna2StatusNotMultipleOf" },
    { Gna2StatusBadFeatLength, "Gna2StatusBadFeatLength" },
    { Gna2StatusDataModeInvalid, "Gna2StatusDataModeInvalid" },
    { Gna2StatusXnnErrorNetLyrNo, "Gna2StatusXnnErrorNetLyrNo" },
    { Gna2StatusXnnErrorNetworkInputs, "Gna2StatusXnnErrorNetworkInputs" },
    { Gna2StatusXnnErrorNetworkOutputs, "Gna2StatusXnnErrorNetworkOutputs" },
    { Gna2StatusXnnErrorLyrOperation, "Gna2StatusXnnErrorLyrOperation" },
    { Gna2StatusXnnErrorLyrCfg, "Gna2StatusXnnErrorLyrCfg" },
    { Gna2StatusXnnErrorLyrInvalidTensorOrder, "Gna2StatusXnnErrorLyrInvalidTensorOrder" },
    { Gna2StatusXnnErrorLyrInvalidTensorDimensions, "Gna2StatusXnnErrorLyrInvalidTensorDimensions" },
    { Gna2StatusXnnErrorInvalidBuffer, "Gna2StatusXnnErrorInvalidBuffer" },
    { Gna2StatusXnnErrorNoFeedback, "Gna2StatusXnnErrorNoFeedback" },
    { Gna2StatusXnnErrorNoLayers, "Gna2StatusXnnErrorNoLayers" },
    { Gna2StatusXnnErrorGrouping, "Gna2StatusXnnErrorGrouping" },
    { Gna2StatusXnnErrorInputBytes, "Gna2StatusXnnErrorInputBytes" },
    { Gna2StatusXnnErrorInputVolume, "Gna2StatusXnnErrorInputVolume" },
    { Gna2StatusXnnErrorOutputVolume, "Gna2StatusXnnErrorOutputVolume" },
    { Gna2StatusXnnErrorIntOutputBytes, "Gna2StatusXnnErrorIntOutputBytes" },
    { Gna2StatusXnnErrorOutputBytes, "Gna2StatusXnnErrorOutputBytes" },
    { Gna2StatusXnnErrorWeightBytes, "Gna2StatusXnnErrorWeightBytes" },
    { Gna2StatusXnnErrorWeightVolume, "Gna2StatusXnnErrorWeightVolume" },
    { Gna2StatusXnnErrorBiasBytes, "Gna2StatusXnnErrorBiasBytes" },
    { Gna2StatusXnnErrorBiasVolume, "Gna2StatusXnnErrorBiasVolume" },
    { Gna2StatusXnnErrorBiasMode, "Gna2StatusXnnErrorBiasMode" },
    { Gna2StatusXnnErrorBiasMultiplier, "Gna2StatusXnnErrorBiasMultiplier" },
    { Gna2StatusXnnErrorBiasIndex, "Gna2StatusXnnErrorBiasIndex" },
    { Gna2StatusXnnErrorPwlSegments, "Gna2StatusXnnErrorPwlSegments" },
    { Gna2StatusXnnErrorPwlData, "Gna2StatusXnnErrorPwlData" },
    { Gna2StatusXnnErrorConvFltBytes, "Gna2StatusXnnErrorConvFltBytes" },
    { Gna2StatusCnnErrorConvFltCount, "Gna2StatusCnnErrorConvFltCount" },
    { Gna2StatusCnnErrorConvFltVolume, "Gna2StatusCnnErrorConvFltVolume" },
    { Gna2StatusCnnErrorConvFltStride, "Gna2StatusCnnErrorConvFltStride" },
    { Gna2StatusCnnErrorConvFltPadding, "Gna2StatusCnnErrorConvFltPadding" },
    { Gna2StatusCnnErrorPoolStride, "Gna2StatusCnnErrorPoolStride" },
    { Gna2StatusCnnErrorPoolSize, "Gna2StatusCnnErrorPoolSize" },
    { Gna2StatusCnnErrorPoolType, "Gna2StatusCnnErrorPoolType" },
    { Gna2StatusGmmBadMeanWidth, "Gna2StatusGmmBadMeanWidth" },
    { Gna2StatusGmmBadMeanOffset, "Gna2StatusGmmBadMeanOffset" },
    { Gna2StatusGmmBadMeanSetoff, "Gna2StatusGmmBadMeanSetoff" },
    { Gna2StatusGmmBadMeanAlign, "Gna2StatusGmmBadMeanAlign" },
    { Gna2StatusGmmBadVarWidth, "Gna2StatusGmmBadVarWidth" },
    { Gna2StatusGmmBadVarOffset, "Gna2StatusGmmBadVarOffset" },
    { Gna2StatusGmmBadVarSetoff, "Gna2StatusGmmBadVarSetoff" },
    { Gna2StatusGmmBadVarsAlign, "Gna2StatusGmmBadVarsAlign" },
    { Gna2StatusGmmBadGconstOffset, "Gna2StatusGmmBadGconstOffset" },
    { Gna2StatusGmmBadGconstAlign, "Gna2StatusGmmBadGconstAlign" },
    { Gna2StatusGmmBadMixCnum, "Gna2StatusGmmBadMixCnum" },
    { Gna2StatusGmmBadNumGmm, "Gna2StatusGmmBadNumGmm" },
    { Gna2StatusGmmBadMode, "Gna2StatusGmmBadMode" },
    { Gna2StatusGmmCfgInvalidLayout, "Gna2StatusGmmCfgInvalidLayout" }
};

GNA2_API enum Gna2Status Gna2StatusGetMessage(
    enum Gna2Status status,
    char * messageBuffer,
    uint32_t messageBufferSize)
{
    const std::function<Gna2Status()> command = [&]()
    {
        GNA::Expect::NotNull(messageBuffer);
        const auto found = Gna2StatusToStringMap.find(status);
        GNA::Expect::True(found != Gna2StatusToStringMap.end(), Gna2StatusIdentifierInvalid);
        GNA::Expect::True(found->second.size() + 1 <= messageBufferSize, Gna2StatusMemorySizeInvalid);
        const auto reqSize = snprintf(messageBuffer, messageBufferSize, "%s", found->second.c_str());
        GNA::Expect::True(reqSize >= 0 && static_cast<unsigned>(reqSize) + 1 <= messageBufferSize,
            Gna2StatusMemorySizeInvalid);
        return Gna2StatusSuccess;
    };
    return GNA::ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2StatusGetMaxMessageLength()
{
    uint32_t maxLen = 0;
    for (const auto & s : Gna2StatusToStringMap)
    {
        maxLen = (std::max)(maxLen, static_cast<uint32_t>(s.second.size()));
    }
    return maxLen + 1;
}

gna_status_t GNA::Gna2GetLegacyStatus(Gna2Status newStatus)
{
    const static std::unordered_map<Gna2Status, gna_status_t, GNA::EnumHash> StatusMap =
    {
        { Gna2StatusSuccess, GNA_SUCCESS },
        { Gna2StatusWarningArithmeticSaturation, GNA_SSATURATE },
        { Gna2StatusWarningDeviceBusy, GNA_DEVICEBUSY },
        { Gna2StatusUnknownError, GNA_UNKNOWN_ERROR },
        { Gna2StatusNotImplemented, GNA_ERR_NOT_IMPLEMENTED },
        { Gna2StatusIdentifierInvalid, GNA_INVALIDHANDLE },
        { Gna2StatusNullArgumentNotAllowed, GNA_NULLARGNOTALLOWED },
        { Gna2StatusNullArgumentRequired, GNA_NULLARGREQUIRED },
        { Gna2StatusResourceAllocationError, GNA_ERR_RESOURCES },
        { Gna2StatusDeviceNotAvailable, GNA_DEVNOTFOUND },
        { Gna2StatusDeviceNumberOfThreadsInvalid, GNA_ERR_INVALID_THREAD_COUNT },
        { Gna2StatusDeviceVersionInvalid, GNA_ERR_INVALID_DEVICE_VERSION },
        { Gna2StatusDeviceQueueError, GNA_ERR_QUEUE },
        { Gna2StatusDeviceIngoingCommunicationError, GNA_IOCTLRESERR },
        { Gna2StatusDeviceOutgoingCommunicationError, GNA_IOCTLSENDERR },
        { Gna2StatusDeviceParameterOutOfRange, GNA_PARAMETEROUTOFRANGE },
        { Gna2StatusDeviceVaOutOfRange, GNA_VAOUTOFRANGE },
        { Gna2StatusDeviceUnexpectedCompletion, GNA_UNEXPCOMPL },
        { Gna2StatusDeviceDmaRequestError, GNA_DMAREQERR },
        { Gna2StatusDeviceMmuRequestError, GNA_MMUREQERR },
        { Gna2StatusDeviceBreakPointHit, GNA_BREAKPOINTPAUSE },
        { Gna2StatusDeviceCriticalFailure, GNA_ERR_DEV_FAILURE },
        { Gna2StatusMemoryAlignmentInvalid, GNA_BADMEMALIGN },
        { Gna2StatusMemorySizeInvalid, GNA_INVALIDMEMSIZE },
        { Gna2StatusMemoryTotalSizeExceeded, GNA_MODELSIZEEXCEEDED },
        { Gna2StatusMemoryBufferInvalid, GNA_INVALID_REQUEST_CONFIGURATION },
        { Gna2StatusRequestWaitError, GNA_WAITFAULT },
        { Gna2StatusActiveListIndicesInvalid, GNA_INVALIDINDICES },
        { Gna2StatusAccelerationModeNotSupported, GNA_CPUTYPENOTSUPPORTED },
        { Gna2StatusModelConfigurationInvalid, GNA_INVALID_MODEL },
        { Gna2StatusNotMultipleOf, GNA_ERR_NOT_MULTIPLE },
        { Gna2StatusBadFeatLength, GNA_BADFEATLENGTH },
        { Gna2StatusXnnErrorNetLyrNo, XNN_ERR_NET_LYR_NO },
        { Gna2StatusXnnErrorNetworkInputs, XNN_ERR_NETWORK_INPUTS },
        { Gna2StatusXnnErrorNetworkOutputs, XNN_ERR_NETWORK_OUTPUTS },
        { Gna2StatusXnnErrorLyrOperation, XNN_ERR_LYR_OPERATION },
        { Gna2StatusXnnErrorLyrCfg, XNN_ERR_LYR_CFG },
        { Gna2StatusXnnErrorLyrInvalidTensorOrder, XNN_ERR_LYR_INVALID_TENSOR_ORDER },
        { Gna2StatusXnnErrorLyrInvalidTensorDimensions, XNN_ERR_LYR_INVALID_TENSOR_DIMENSIONS },
        { Gna2StatusXnnErrorInvalidBuffer, XNN_ERR_INVALID_BUFFER },
        { Gna2StatusXnnErrorNoFeedback, XNN_ERR_NO_FEEDBACK },
        { Gna2StatusXnnErrorNoLayers, XNN_ERR_NO_LAYERS },
        { Gna2StatusXnnErrorGrouping, XNN_ERR_GROUPING },
        { Gna2StatusXnnErrorInputBytes, XNN_ERR_INPUT_BYTES },
        { Gna2StatusXnnErrorInputVolume, XNN_ERR_INPUT_VOLUME },
        { Gna2StatusXnnErrorOutputVolume, XNN_ERR_OUTPUT_VOLUME },
        { Gna2StatusXnnErrorIntOutputBytes, XNN_ERR_INT_OUTPUT_BYTES },
        { Gna2StatusXnnErrorOutputBytes, XNN_ERR_OUTPUT_BYTES },
        { Gna2StatusXnnErrorWeightBytes, XNN_ERR_WEIGHT_BYTES },
        { Gna2StatusXnnErrorWeightVolume, XNN_ERR_WEIGHT_VOLUME },
        { Gna2StatusXnnErrorBiasBytes, XNN_ERR_BIAS_BYTES },
        { Gna2StatusXnnErrorBiasVolume, XNN_ERR_BIAS_VOLUME },
        { Gna2StatusXnnErrorBiasMode, XNN_ERR_BIAS_MODE },
        { Gna2StatusXnnErrorBiasMultiplier, XNN_ERR_BIAS_MULTIPLIER },
        { Gna2StatusXnnErrorBiasIndex, XNN_ERR_BIAS_INDEX },
        { Gna2StatusXnnErrorPwlSegments, XNN_ERR_PWL_SEGMENTS },
        { Gna2StatusXnnErrorPwlData, XNN_ERR_PWL_DATA },
        { Gna2StatusXnnErrorConvFltBytes, XNN_ERR_CONV_FLT_BYTES },
        { Gna2StatusCnnErrorConvFltCount, CNN_ERR_CONV_FLT_COUNT },
        { Gna2StatusCnnErrorConvFltVolume, CNN_ERR_CONV_FLT_VOLUME },
        { Gna2StatusCnnErrorConvFltStride, CNN_ERR_CONV_FLT_STRIDE },
        { Gna2StatusCnnErrorConvFltPadding, CNN_ERR_CONV_FLT_PADDING },
        { Gna2StatusCnnErrorPoolStride, CNN_ERR_POOL_STRIDE },
        { Gna2StatusCnnErrorPoolSize, CNN_ERR_POOL_SIZE },
        { Gna2StatusCnnErrorPoolType, CNN_ERR_POOL_TYPE },
        { Gna2StatusGmmBadMeanWidth, GMM_BADMEANWIDTH },
        { Gna2StatusGmmBadMeanOffset, GMM_BADMEANOFFSET },
        { Gna2StatusGmmBadMeanSetoff, GMM_BADMEANSETOFF },
        { Gna2StatusGmmBadMeanAlign, GMM_BADMEANALIGN },
        { Gna2StatusGmmBadVarWidth, GMM_BADVARWIDTH },
        { Gna2StatusGmmBadVarOffset, GMM_BADVAROFFSET },
        { Gna2StatusGmmBadVarSetoff, GMM_BADVARSETOFF },
        { Gna2StatusGmmBadVarsAlign, GMM_BADVARSALIGN },
        { Gna2StatusGmmBadGconstOffset, GMM_BADGCONSTOFFSET },
        { Gna2StatusGmmBadGconstAlign, GMM_BADGCONSTALIGN },
        { Gna2StatusGmmBadMixCnum, GMM_BADMIXCNUM },
        { Gna2StatusGmmBadNumGmm, GMM_BADNUMGMM },
        { Gna2StatusGmmBadMode, GMM_BADMODE },
        { Gna2StatusGmmCfgInvalidLayout, GMM_CFG_INVALID_LAYOUT },
    };
    return StatusMap.at(newStatus);
}

Gna2DeviceVersion GNA::Gna2GetVersionForLegacy(gna_device_version legacyVersion)
{
    const static std::unordered_map<gna_device_version, Gna2DeviceVersion, GNA::EnumHash> DeviceVersionMapInverted =
    {
        {GNA_GMM, Gna2DeviceVersionGMM },
        {GNA_0x9, Gna2DeviceVersion0_9 },
        {GNA_1x0, Gna2DeviceVersion1_0 },
        {GNA_2x0, Gna2DeviceVersion2_0 },
        {GNA_3x0, Gna2DeviceVersion3_0 },
        {GNA_EMBEDDED_1x0, Gna2DeviceVersionEmbedded1_0 },
        {GNA_EMBEDDED_2x1, Gna2DeviceVersionEmbedded2_1 },
        {GNA_EMBEDDED_3x0, Gna2DeviceVersionEmbedded3_0 },
        {GNA_EMBEDDED_3x1, Gna2DeviceVersionEmbedded3_1 },
        {GNA_SOFTWARE_EMULATION, Gna2DeviceVersionSoftwareEmulation }
    };
    return DeviceVersionMapInverted.at(legacyVersion);
}
