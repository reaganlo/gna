/*
 INTEL CONFIDENTIAL
 Copyright 2020 Intel Corporation.

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

#include "HwModuleInterface.hpp"

#include "ConvolutionalFunctions.h"
#include "ConvolutionalFunctions2D.h"
#include "DataMode.h"
#include "LinuxHwModuleInterface.hpp"
#include "PoolingFunctions2D.h"
#include "WindowsHwModuleInterface.hpp"

#include "gna-api-types-xnn.h"

#if 1 == GNA_HW_LIB_ENABLED
#include "GNA_ArchCPkg.h"
#endif

#include <cstdint>
#include <memory>

#undef GNA_HW_MODULE_CLASS
#if defined(_WIN32)
#   define GNA_HW_MODULE_CLASS WindowsHwModuleInterface
#else // GNU/Linux / Android / ChromeOS
#   define GNA_HW_MODULE_CLASS LinuxHwModuleInterface
#endif

using namespace GNA;

HwUarchParams::HwUarchParams(struct GNA3_AdaptHW const& source)
#if 1 == GNA_HW_LIB_ENABLED
    :
    Valid{source.Valid},
    KWG{source.KWG},
    KWGIter{source.KWGIter},
    uT{source.uT},
    KMemBase{source.KMemBase},
    CMemBase{source.CMemBase},
    PMemBase{source.PMemBase}
{
}
#else
{
    UNREFERENCED_PARAMETER(source);
}
#endif

std::unique_ptr<HwModuleInterface const> HwModuleInterface::Create(char const* moduleName)
{
    Expect::NotNull(moduleName);
    Expect::False(std::string(moduleName).empty(), Gna2StatusAccelerationModeNotSupported);
    try
    {
        return std::make_unique<GNA_HW_MODULE_CLASS const>(moduleName);
    }
    catch (GnaException & e)
    {
        Log->Warning("HwModule library load failed.");
        throw e;
    }
}

HwUarchParams HwModuleInterface::GetCnnParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                              const DataMode& outputMode, bool is1D) const
{
    Expect::NotNull(reinterpret_cast<const void*>(CreateLD),
                    Gna2StatusAccelerationModeNotSupported);
    Expect::NotNull(cnnIn);
    Expect::NotNull(poolingIn);

    if (is1D)
    {
        return Get1DParams(cnnIn, poolingIn, outputMode);
    }
    return Get2DParams(cnnIn, poolingIn, outputMode);
}

HwUarchParams HwModuleInterface::Get1DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                             const DataMode& outputMode) const
{
#if 1 == GNA_HW_LIB_ENABLED
    auto const LD_2DCNN = CreateLD();

    LD_2DCNN->IFV.N = 1; // Must set to 1, for IFVs
    LD_2DCNN->IFV.W = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_W));
    LD_2DCNN->IFV.H = 1;
    LD_2DCNN->IFV.C = 1;
    LD_2DCNN->IFV.Prec = static_cast<GNA3_Prec_t>(cnnIn->Input->Mode.Size);
    // Kernels @ Setting 2DCNNc Parameters
    LD_2DCNN->Op = GNA3_OP_1DCNN;
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.NConvFilters = static_cast<uint16_t>(cnnIn->Filters->Count);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.KPrec = static_cast<GNA3_Prec_t>(cnnIn->Filters->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.NConvFilterElements = static_cast<uint16_t>(cnnIn->Filters->at(GNA_DIM_W));
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.InputConvStride = static_cast<uint16_t>(cnnIn->Stride->at(GNA_DIM_W));
    // BIAS @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.BPrec = static_cast<GNA3_Prec_t>(cnnIn->Biases->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.BType = GNA3_BIASperKERNEL; //other modes not supported
    // Pooling @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PType = poolingIn == nullptr || poolingIn->Mode == KernelPoolingModeNone
                                                 ? GNA3_POOL_DIS
                                                 : static_cast<GNA3_PoolType_t>(poolingIn->Mode);
    if (LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PType != GNA3_POOL_DIS)
    {
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PWin = static_cast<uint8_t>(poolingIn->Window->at(GNA_DIM_W));
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PStr = static_cast<uint8_t>(poolingIn->Stride->at(GNA_DIM_W));
    }
    else
    {
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PWin = 0;
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PStr = 0;
    }
    // Activation @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.ACTx = static_cast<GNA3_Prec_t>(outputMode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.NSegs = 0;

    auto const validationResult = FillLD(LD_2DCNN);
    auto adaptHW = GNA3_AdaptHW_t{LD_2DCNN->AdaptHW};
    FreeLD(LD_2DCNN);

    if (!validationResult)
    {
        adaptHW.Valid = false;
    }

    return HwUarchParams{adaptHW};
#else
    UNREFERENCED_PARAMETER(cnnIn);
    UNREFERENCED_PARAMETER(poolingIn);
    UNREFERENCED_PARAMETER(outputMode);
    throw GnaException(Gna2StatusNotImplemented);
#endif
}

HwUarchParams HwModuleInterface::Get2DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                             const DataMode& outputMode) const
{
#if 1 == GNA_HW_LIB_ENABLED
    auto const LD_2DCNN = CreateLD();

    LD_2DCNN->IFV.N = 1; // Must set to 1, for IFVs
    LD_2DCNN->IFV.H = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_H));
    LD_2DCNN->IFV.W = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_W));
    LD_2DCNN->IFV.C = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_D));
    LD_2DCNN->IFV.Prec = static_cast<GNA3_Prec_t>(cnnIn->Input->Mode.Size);
    // Kernels @ Setting 2DCNNc Parameters
    LD_2DCNN->Op = GNA3_OP_2DCNNc;
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KNum = static_cast<uint16_t>(cnnIn->Filters->Count);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KPrec = static_cast<GNA3_Prec_t>(cnnIn->Filters->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KDim.H = static_cast<uint16_t>(cnnIn->Filters->at(GNA_DIM_H));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KDim.W = static_cast<uint16_t>(cnnIn->Filters->at(GNA_DIM_W));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CStr.H = static_cast<uint16_t>(cnnIn->Stride->at(GNA_DIM_H));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CStr.W = static_cast<uint16_t>(cnnIn->Stride->at(GNA_DIM_W));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CZPad.H = static_cast<uint16_t>(cnnIn->Padding->at(GNA_DIM_H));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CZPad.W = static_cast<uint16_t>(cnnIn->Padding->at(GNA_DIM_W));
    // BIAS @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.BPrec = static_cast<GNA3_Prec_t>(cnnIn->Biases->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.BType = GNA3_BIASperKERNEL; //other modes not supported
    // Pooling @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PType = poolingIn == nullptr || poolingIn->Mode == KernelPoolingModeNone
                                                  ? GNA3_POOL_DIS
                                                  : static_cast<GNA3_PoolType_t>(poolingIn->Mode);
    if (LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PType != GNA3_POOL_DIS)
    {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.H = static_cast<uint16_t>(poolingIn->Window->at(GNA_DIM_H));
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.W = static_cast<uint16_t>(poolingIn->Window->at(GNA_DIM_W));
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.H = static_cast<uint16_t>(poolingIn->Stride->at(GNA_DIM_H));
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.W = static_cast<uint16_t>(poolingIn->Stride->at(GNA_DIM_W));
    }
    else
    {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.H = 0;
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.W = 0;
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.H = 0;
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.W = 0;
    }
    // Activation @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.ACTx = static_cast<GNA3_Prec_t>(outputMode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.NSegs = 0;

    auto const validationResult = FillLD(LD_2DCNN);
    auto adaptHW = GNA3_AdaptHW_t{LD_2DCNN->AdaptHW};
    FreeLD(LD_2DCNN);

    if (!validationResult)
    {
        adaptHW.Valid = false;
    }

    return HwUarchParams{adaptHW};
#else
    UNREFERENCED_PARAMETER(cnnIn);
    UNREFERENCED_PARAMETER(poolingIn);
    UNREFERENCED_PARAMETER(outputMode);
    throw GnaException(Gna2StatusNotImplemented);
#endif
}

void HwModuleInterface::Validate() const
{
    Expect::NotNull(reinterpret_cast<const void*>(CreateLD));
    Expect::NotNull(reinterpret_cast<const void*>(FreeLD));
    Expect::NotNull(reinterpret_cast<const void*>(FillLD));
    Log->Message("HwModule library loaded successfully.\n");
}
