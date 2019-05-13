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

#ifndef __GNA2_COMMON_IMPL_H
#define __GNA2_COMMON_IMPL_H

#include "gna-api-status.h"
#include "../gna-api/gna2-common-api.h"

#include <stdint.h>
#include <unordered_map>

namespace GNA
{

typedef enum Gna2DeviceVersion DeviceVersion;

DeviceVersion const DefaultDeviceVersion = GNA2_DEFAULT_DEVICE_VERSION;

typedef enum Gna2Status ApiStatus;

constexpr uint32_t const Gna2DisabledU32 = (uint32_t)GNA2_DISABLED;

constexpr int32_t const Gna2DisabledI32 = (int32_t)GNA2_DISABLED;

constexpr uint32_t const Gna2DefaultU32 = (uint32_t)GNA2_DEFAULT;

constexpr int32_t const Gna2DefaultI32 = (int32_t)GNA2_DEFAULT;

constexpr uint32_t const Gna2NotSupportedU32 = (uint32_t)GNA2_NOT_SUPPORTED;

constexpr int32_t const Gna2NotSupportedI32 = (int32_t)GNA2_NOT_SUPPORTED;

/* Workaround for old compilers that do not handle enums as map keys */
struct EnumHash
{
    template<typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};


const std::unordered_map<Gna2Status, gna_status_t, EnumHash> StatusMap =
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

}

#endif //ifndef __GNA2_COMMON_IMPL_H
