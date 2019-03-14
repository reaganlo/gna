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

#pragma once

#include <string.h>

#include "common.h"

#include "Address.h"
#include "GnaConfig.h"
#include "HardwareCapabilities.h"

namespace GNA
{

using AddrGmmCfg = Address<GMM_CONFIG *>;
using AddrGmmCfgC = Address<GMM_CONFIG * const>;

// Available Xnn Layer Descriptor parameters for all hw versions
typedef enum _GmmParameterType
{
    fvaddr,
    fvoffset,
    fvwidth,
    mode,
    read_elimination,
    calculation_mode,
    numfv,
    vlength,
    mvaddr,
    mvwidth,
    mvsoffset,
    vvaddr,
    vvwidth,
    vvsoffset,
    gcaddr,
    gcwidth,
    gcsoffset,
    maxlsscore,
    maxlswidth,
    nummcpg,
    gmmtelst,
    numgmms,
    asladdr,
    astlistlen,
    gmmscrwdth,
    gmmscradd,
    gmmscrlen,
} GmmParameterType;

// Available Xnn Layer Descriptor parameters for all hw versions
typedef enum _XnnParameterType
{
    op,                 // 0x00 Type of xNN operation to be scored (NN_OP_TYPE)
    active_list_enabled,// 0x00 Affine AL bit
    flags,              // 0x01 flags for debug purposes only
    act_fn_precision,   // GNA 0.9+
                        //  0x01:02:02 Activation function is disabled (0b0) or enabled (0b1)
                        // GNA 3.0+
                        //  0x01:05:04 Activation function precision
                        //      00 - disabled
                        //      01 - 8bit
                        //      10 - 16bit
                        //      11 - 32bit
    input_element_precision,// GNA 3.0+
                        //  0x01:03:02 Input element precision 3.0
                        //      00 - disabled
                        //      01 - 16bit
                        //      10 - 8bit
                        //      11 - reserved
    weight_size,        // GNA 0.9+
                        //  0x01:00:01 Weight element size:
                        //      00 - 16-bit element, Dens Const format
                        //      01 - 8-bit element, Rich Const format
                        // GNA 3.0+
                        //  0x01:00:01 Weight/kernel element size/precision:
                        //      00 - const
                        //      01 - 16-bit element
                        //      10 - 8-bit element
                        //      11 - reserved
    pool_param,         // 0x01:03:04 No Pool (0b00), MaxPool (0b01), AvaragePool (0b10), Reserved (0b11). Applicable in CNN layers only.
    n_in_elems,         // 0x02; 0x03 Total number of input elements

    n_out_elems,        // 0x04; 0x05 Number of output elements [1 - (2^16-1)]
    cnn_n_out_p_flt,    // 0x04; 0x05 CNN Number of output elements per Filter in full iterations

    n_groups,           // 0x06; 0x06 Number of input groups used
    cnn_n_flt_last,     // 0x06; 0x06 CNN Number of filters in buffer in last iteration [4,8,12,16)]

    n_iters,            // 0x07; 0x07 Blocking size used to fit size of input buffer
    cnn_pool_stride,    // 0x07; 0x07 CNN Pool Stride [1-6]

    n_elems_last,       // 0x08; 0x09 Number of input elements in last iteration per group
    cnn_n_flt_stride,   // 0x08; 0x09 CNN Input-filter stride - Number of input elements for convolution operation [1-768]

    rnn_n_fb_iters,     // 0x0a; 0x0a Number of iterations in feedback stage
    cnn_pool_size,      // 0x0a; 0x0a CNN Size of Pool [1-6]

    bias_precision,     // Bias element precision

    rnn_n_elems_first,  // 0x0c; 0x0d Number of elements in first feedback iteration
    cnn_n_flts,         // 0x0c; 0x0d CNN Number of convolution filters [4 - (2^16 - 4)], %4

    rnn_n_elems_last,   // 0x0e; 0x0f Number of elements in last feedback iteration
    cnn_n_flt_iters,    // 0x0e; 0x0f CNN Number of iterations for all convolution filters

    pwl_n_segs,         // 0x10; 0x10 Number of activation function segments

    act_list_n_elems,   // 0x12; 0x13 Number of output elements in output active list enabled mode
    cpy_n_elems,        // 0x12; 0x13 Number of elements copied in copy OP operation [8 - (2^16 - 8)], %8
    cnn_flt_size,       // 0x12; 0x13 CNN convolution filter size (elements per filter) [48 - 768], %8
    bias_grp_cnt,       // 0x12; 0x13 Grouping of the bias array [1-8]

    cnn_n_flts_iter,    // 0x14; 0x15 CNN Number of filters in input buffer in full iterations [4,8,12,16]
    bias_grp_value,     // 0x14; 0x15 Current column selected [0-7]

    cnn_n_flt_outs,     // 0x16; 0x17 CNN Number of output elements per Filter after conv., before pooling
    cnn_flt_bf_sz_iter, // 0x18; 0x19 CNN filter buffer size per (non-last) iteration (B) [1-InBufSize/2]
    cnn_flt_bf_sz_last, // 0x1A; 0x1B CNN filter buffer size in last iteration (B) [1-InBufSize/2]

    in_buffer,          // 0x20; 0x23 Pointer to input array [2B elements]
    gmm_descriptor,     // 0x20; 0x23 Pointer GMM layer descriptor

    out_buffer,  // 0x24; 0x27 Pointer to 2B output array after pwl act. fn. [2B elements]
    out_sum_buffer,     // 0x28; 0x2B Pointer to 4B intermediate output sum array. [4B elements]
    rnn_out_fb_buffer,  // 0x2C; 0x2f Pointer to output FB array

    weight_buffer,      // 0x30; 0x33 Pointer to weights/kernel array

    bias_buffer,   // 0x34; 0x37 Pointer to const and weight scale array. [4B elements or 1B scale +3B res.]

    act_list_buffer,    // 0x38; 0x3b Active outputs list pointer [4B elements]
    bias_grp_buffer,    // 0x38; 0x3b Bias grouping array pointer [4B elements]

    pwl_seg_def_buffer, // 0x3c; 0x3f Pointer to array that holds the activation function section definition [8B elements]


    cnn2d_in_dim_d,     // CNN2D Input Volume Dimension - Depth
    cnn2d_in_dim_w,     // CNN2D Input Volume Dimension - Width
    cnn2d_in_dim_h,     // CNN2D Input Volume Dimension - Height
    cnn2d_conv_kernel_w,// CNN2D Convolution Kernel Dimension - Width
    cnn2d_conv_kernel_h,// CNN2D Convolution Kernel Dimension - Height
    cnn2d_conv_out_w,   // CNN2D Convolution Output Dimension - Width
    cnn2d_conv_out_h,   // CNN2D Convolution Output Dimension - Height
    cnn2d_conv_stride_w,// CNN2D Convolution Stride Dimension - Width
    cnn2d_conv_stride_h, // CNN2D ConvolutionStride Dimension - Height
    cnn2d_pool_out_w,   // CNN2D Pooling Output Dimension - Width
    cnn2d_pool_out_h,   // CNN2D Pooling Output Dimension - Height
    cnn2d_pool_stride_w,// CNN2D Pooling Stride Dimension - Width
    cnn2d_pool_stride_h,// CNN2D Pooling Stride Dimension - Height
    cnn2d_pool_window_w,// CNN2D Pooling Window Dimension - Width
    cnn2d_pool_window_h,// CNN2D Pooling Window Dimension - Height
    cnn2d_padding_w,    // CNN2D Zero-padding Dimension - Width
    cnn2d_padding_h,    // CNN2D Zero-padding Dimension - Height
    cnn2d_addaptive,    // CNN2D Adaptive Hardware [uArch] knobs // TODO:3:CNN2D: Adaptive Hardware PCR?
    cnn2d_kernel_wg,    // CNN2D Convolution Kernel work group size
    cnn2d_kernel_iter,  // CNN2D Convolution Kernel work group iterations
    cnn2d_kernel_scalar,// CNN2D Convolution Kernel constant scalar
    cnn2d_bias_mode,    // CNN2D Convolution Bias Mode

} XnnParameterType;

typedef const std::map<const uint32_t, const uint8_t> ParamTranslator;

class XnnParameter
{
public:
    XnnParameter(uint32_t offsetIn, uint32_t size) :
        Size { size },
        offset { offsetIn },
        address {},
        translator{nullptr}
    {
    }

    /**
     * Creates bit flags typeXnnParameter
     * @bitOffset  0-based index of first bit field bit
     * @bitCount   number of bits in bit field
     */
    XnnParameter(uint32_t offsetIn, uint32_t size, uint8_t bitOffsetIn, uint8_t bitCountIn,
        const ParamTranslator& translatorIn) :
        Size{size},
        offset{offsetIn},
        address{},
        bitOffset{bitOffsetIn},
        bitCount{bitCountIn},
        translator{std::make_shared<const ParamTranslator>(translatorIn)}
    {
        Expect::InRange<uint8_t>(bitOffset, 31, GNA_ERR_INVALID_DATA_MODE);
        Expect::InRange<uint8_t>(bitCount, 1, 32, GNA_ERR_INVALID_DATA_MODE);
        Expect::NotNull(translator.get());
    }

    XnnParameter(BaseAddress descriptorAddress, uint32_t descriptorOffset,
                const XnnParameter& param, const GetHwOffset getHwOffset) :
        Size { param.Size },
        offset { param.offset },
        address { descriptorAddress + offset },
        absoluteOffset{ descriptorOffset + offset },
        bitOffset { param.bitOffset },
        bitCount { param.bitCount },
        translator { param.translator },
        getBufferOffset { getHwOffset }
    {
    }

    void operator=(const uint8_t& value)
    {
        set(value);
    }

    void operator=(const gna_data_mode mode)
    {
        set(mode);
    }

    void operator=(const gna_bias_mode mode)
    {
        set(mode);
    }

    void operator=(const uint32_t& value)
    {
        switch (Size)
        {
        case 1:
            set(static_cast<uint8_t>(value));
        break;
        case 2:
            set(static_cast<uint16_t>(value));
        break;
        default:
            set(static_cast<uint32_t>(value));
        break;
        }
    }

    // sets value as absolute offset
    void operator=(const BaseAddress& buffer)
    {
        Expect::True(4 == Size && 0 == bitCount, GNA_UNKNOWN_ERROR);
        *address.Get<uint32_t>() = getBufferOffset(buffer);
    }

    uint8_t* operator&() const
    {
        return address.Get<uint8_t>();
    }

    // returns parameter value
    uint32_t Get() const
    {
        switch (Size)
        {
        case 1:
            return static_cast<uint32_t>(*address.Get<uint8_t>());
        break;
        case 2:
            return static_cast<uint32_t>(*address.Get<uint16_t>());
        break;
        default:
            return static_cast<uint32_t>(*address.Get<uint32_t>());
        break;
        }
    }

    uint32_t GetOffset() const
    {
        return absoluteOffset;
    }

    uint32_t    Size;
private:
    uint32_t    offset; // Parameter offset relative to LayerDescriptor base
    BaseAddress address;            // global parameter address
    uint32_t absoluteOffset = 0;    // absolute parameters offset from model memory base
    uint8_t bitOffset = 0;
    uint8_t bitCount = 0;
    const std::shared_ptr<const ParamTranslator> translator;
    GetHwOffset getBufferOffset;

     // sets parameter value
    template<typename T>
    void set(const T& value)
    {
        if (0 == bitCount)
        {
            *address.Get<T>() = value;
        }
        else // set bit flags
        {
            uint8_t newValue = translator->at(value);
            uint8_t mask = 0xFF;        // FF
            mask = mask >> (8 - bitCount);// 2: 03
            newValue &= mask;      // fit value  (03)
            mask = mask << bitOffset;   // 3: 18 place mask at desired position
            newValue <<= bitOffset;     // 3: 18 place mask at desired position
            mask = ~mask;               // E7 neg mask
            *address.Get<uint8_t>() &= mask; // clear old value E7
            *address.Get<uint8_t>() |= newValue; // set new value FF
        }
    }
};


class LayerDescriptor
{
public:
    // Gets default size of descriptor
    inline static size_t GetSize()
    {
        return getSize(GNA_DEFAULT_VERSION);
    }

    // Gets total size of all layers' descriptors for given hw
    inline static uint32_t GetSize(const uint32_t layerCount,
        const gna_device_version hwId = GNA_DEFAULT_VERSION)
    {
        return static_cast<const uint32_t>(getSize(hwId) * layerCount);
    }

    LayerDescriptor() = delete;

    LayerDescriptor(const LayerDescriptor&) = default;

    LayerDescriptor(const BaseAddress memoryBase, const BaseAddress& address,
                    const HardwareCapabilities& hwCaps);

    LayerDescriptor(const LayerDescriptor& base, AddrGmmCfg gmmDescriptor, GetHwOffset getHwOffsetIn);

    ~LayerDescriptor() = default;

    void Forward(AddrGmmCfg gmmDescriptor)
    {
        address = address + static_cast<uint32_t>(Size);
        offset = offset + static_cast<uint32_t>(Size);
        GmmDescriptor = gmmDescriptor;
    }

    bool HasParameter(const XnnParameterType paramType) const
    {
        return 1 == xnnReferenceParams->count(paramType);
    }

    // return XnnParameter copy for GMM data manipulation
    XnnParameter operator[](const GmmParameterType paramType) const
    {
        auto gmmOffset = GmmDescriptor.GetOffset(memoryBase);
        return XnnParameter(GmmDescriptor, gmmOffset, gmmReferenceParams->at(paramType), getHwOffset);
    }

    // return XnnParameter copy for xNN data manipulation
    XnnParameter operator[](const XnnParameterType paramType) const
    {
        return XnnParameter(address, offset, xnnReferenceParams->at(paramType), getHwOffset);
    }

    BaseAddress GetMemAddress() const
    {
        return address;
    }

    uint32_t GetOffset() const
    {
        return address.GetOffset(memoryBase);
    }

    size_t Size;
    const HardwareCapabilities& HwCapabilities;
    AddrGmmCfg GmmDescriptor;

private:
    static size_t getSize(const gna_device_version hwId);
    static const std::map<const XnnParameterType, const XnnParameter>& getParameterMap(const gna_device_version hwId);

    LayerDescriptor(const AddrGmmCfg gmmConfig, const size_t size, const HardwareCapabilities& hwCaps,
        const BaseAddress memoryBaseIn, BaseAddress descriptorBaseIn,
        const std::map<const XnnParameterType, const XnnParameter>& paramsIn,
        GetHwOffset getHwOffsetIn);

    BaseAddress memoryBase;
    BaseAddress address;     // LayerDescriptor memory address
    uint32_t offset;      // absolute offset of LayerDescriptor from model memory base
    const std::map<const XnnParameterType, const XnnParameter>* xnnReferenceParams;
    const std::map<const GmmParameterType, const XnnParameter>* gmmReferenceParams;
    GetHwOffset getHwOffset;
};

}
