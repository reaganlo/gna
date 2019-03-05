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

#include "LayerDescriptor.h"

using namespace GNA;

size_t LayerDescriptor::getSize(const gna_device_version hwId)
{
    static const std::map<const gna_device_version, const size_t> sizeMap =
    {

        {GNA_CNL, 128},
        {GNA_GLK, 128},
        {GNA_ICL, 128},
        {GNA_TGL, 128},
        {GNA_ADL, 128},
        {GNA_SUE_CREEK, 128},
        {GNA_JELLYFISH, 128},
        {GNA_ACE_EMBEDDED, 128}, // TODO:3: verify the actual size
        {GNA_ACE_ANNA, 128},
    };
    return sizeMap.at(hwId);
}

static const std::map<const GmmParameterType, const XnnParameter> GmmDescriptorGNA =
{
    { fvaddr, { 0x00, 4 }},
    { fvoffset, {0x04, 4}},
    { fvwidth, {0x08, 4}},
    { mode, {0x0c, 4 }},
    { read_elimination, {0x0c, 4, 0, 1,
                            {
                                {GMM_NORMAL_OPERATION, static_cast<uint8_t>(0)},
                                {GMM_READ_ELIMINATION_ENABLED, static_cast<uint8_t>(1)},
                            }}},
    { calculation_mode, {0x0c, 4, 1, 2,
                            {
                                {GMM_L2_DISTANCE, static_cast<uint8_t>(0)},
                                {GMM_L1_DISTANCE, static_cast<uint8_t>(1)},
                                {GMM_LINF_DISTANCE, static_cast<uint8_t>(2)},
                            }}},
    { numfv, {0x10, 4}},
    { vlength, {0x14, 4}},
    { mvaddr, {0x18, 4}},
    { mvwidth, {0x20, 4}},
    { mvsoffset, {0x28, 4}},
    { vvaddr, {0x30, 4}},
    { vvwidth, {0x38, 4}},
    { vvsoffset, {0x40, 4}},
    { gcaddr, {0x44, 4}},
    { gcwidth, {0x4c, 4}},
    { gcsoffset, {0x50, 4}},
    { maxlsscore, {0x54, 4}},
    { maxlswidth, {0x58, 4}},
    { nummcpg, {0x5C, 4}},
    { gmmtelst, {0x60, 4}},
    { numgmms, {0x64, 4}},
    { asladdr, {0x68, 4}},
    { astlistlen, {0x70, 4}},
    { gmmscrwdth, {0x74, 4}},
    { gmmscradd, {0x78, 4}},
    { gmmscrlen, {0x7c, 4}},
};


static const std::map<const XnnParameterType, const XnnParameter> XnnDescriptorGNA_1 =
{
    { op,{ 0x00, 1 }},
    {flags, { 0x01, 1 }},
    {act_fn_precision, { 0x01, 1, 2, 1,
        {
            {GNA_DATA_ACTIVATION_DISABLED, static_cast<uint8_t>(0)},
            {GNA_INT32, static_cast<uint8_t>(0)},
            {GNA_INT16, static_cast<uint8_t>(1)},
        }}},
    {weight_size, { 0x01, 1, 0, 2,
        {
            {GNA_INT8, static_cast<uint8_t>(1)},
            {GNA_INT16, static_cast<uint8_t>(0)},
        }}},
    {pool_param, { 0x01, 1, 3, 2,
         {
            {INTEL_NO_POOLING, static_cast<uint8_t>(0)},
            {INTEL_MAX_POOLING, static_cast<uint8_t>(1)},
            {INTEL_SUM_POOLING, static_cast<uint8_t>(2)},
        }}},
    {n_in_elems, { 0x02, 2 }},
    {n_out_elems, { 0x04, 2 }},
    {cnn_n_out_p_flt, { 0x04, 2 }},
    {n_groups, { 0x06, 1 }},
    {cnn_n_flt_last, { 0x06, 1 }},
    {n_iters, { 0x07, 1 }},
    {cnn_pool_stride, { 0x07, 1 }},
    {n_elems_last, { 0x08, 2 }},
    {cnn_n_flt_stride, { 0x08, 2 }},
    {rnn_n_fb_iters, { 0x0a, 1 }},
    {cnn_pool_size, { 0x0a, 1 }},
    {rnn_n_elems_first, { 0x0c, 2 }},
    {cnn_n_flts, { 0x0c, 2 }},
    {rnn_n_elems_last, { 0x0e, 2 }},
    {cnn_n_flt_iters, { 0x0e, 2 }},
    {pwl_n_segs, { 0x10, 1 }},
    {act_list_n_elems, { 0x12, 2 }},
    {cpy_n_elems, { 0x12, 2 }},
    {cnn_flt_size, { 0x12, 2 }},
    {bias_grp_cnt, { 0x12, 2 }},
    {cnn_n_flts_iter, { 0x14, 2 }},
    {bias_grp_value, { 0x14, 2 }},
    {cnn_n_flt_outs, { 0x16, 2 }},
    {cnn_flt_bf_sz_iter, { 0x18, 2 }},
    {cnn_flt_bf_sz_last, { 0x1A, 2 }},
    {in_buffer, { 0x20, 4 }},
    {gmm_descriptor, { 0x20, 4 }},
    {out_buffer, { 0x24, 4 }},
    {out_sum_buffer, { 0x28, 4 }},
    {rnn_out_fb_buffer, { 0x2C, 4 }},
    {weight_buffer, { 0x30, 4 }},
    {bias_buffer, { 0x34, 4 }},
    {act_list_buffer, { 0x38, 4 }},
    {bias_grp_buffer, { 0x38, 4 }},
    {pwl_seg_def_buffer, { 0x3c, 4 }},
};


static const std::map<const XnnParameterType, const XnnParameter> XnnDescriptorGNA_3 =
{
    {op, { 0x00, 1 }},
    {active_list_enabled, { 0x00, 1, 0, 1,
        {
            { 0, static_cast<uint8_t>(0)},
            { 1, static_cast<uint8_t>(1) }
        }}},
    {flags, { 0x01, 1 }},
    {act_fn_precision, { 0x01, 1, 4, 2,
        {
            {GNA_DATA_ACTIVATION_DISABLED, static_cast<uint8_t>(0)},
            {GNA_INT8, static_cast<uint8_t>(1)},
            {GNA_INT16, static_cast<uint8_t>(2)},
            {GNA_INT32, static_cast<uint8_t>(3)}
        }}},
    {input_element_precision, { 0x01, 1, 2, 2,
        {
            {GNA_DATA_DISABLED, static_cast<uint8_t>(0)},
            {GNA_INT8, static_cast<uint8_t>(1)},
            {GNA_INT16, static_cast<uint8_t>(2)},
        }}},
    {weight_size, { 0x01, 1, 0, 2,
        {
            {GNA_DATA_CONSTANT_SCALAR, static_cast<uint8_t>(0)},
            {GNA_INT8, static_cast<uint8_t>(1)},
            {GNA_INT16, static_cast<uint8_t>(2)}
        }}},
    {n_in_elems, { 0x02, 2 }},
    {n_out_elems, { 0x04, 2 }},
    {cnn_n_out_p_flt, { 0x04, 2 }},
    {n_groups, { 0x06, 1 }},
    {cnn_n_flt_last, { 0x06, 1 }},
    {n_iters, { 0x07, 1 }},
    {cnn_pool_stride, { 0x07, 1 }},
    {n_elems_last, { 0x08, 2 }},
    {cnn_n_flt_stride, { 0x08, 2 }},
    {rnn_n_fb_iters, { 0x0a, 1 }},
    {cnn_pool_size, { 0x0a, 1 }},
    {bias_precision, { 0x0b, 1, 0, 3,
                         {
                             {GNA_DATA_CONSTANT_SCALAR, static_cast<uint8_t>(0) },
                             {GNA_INT8, static_cast<uint8_t>(1) },
                             {GNA_INT16, static_cast<uint8_t>(2) },
                             {GNA_INT32, static_cast<uint8_t>(3) },
                             {GNA_DATA_RICH_FORMAT, static_cast<uint8_t>(7) },
        }}}, //When using ‘Rich-Format’, Constants values are bounded to INT-32 precision.
        //Therefore, NNFlags::BPRC should have no impact on GNA-HW. However this is not true in newest FPGA image.
    { pool_param, { 0x0B, 1, 6, 2,
                      {
                          { INTEL_NO_POOLING, static_cast<uint8_t>(0) },
                          { INTEL_MAX_POOLING, static_cast<uint8_t>(1) },
                          { INTEL_SUM_POOLING, static_cast<uint8_t>(2) },
                      } } },
    {rnn_n_elems_first, { 0x0c, 2 }},
    {cnn_n_flts, { 0x0c, 2 }},
    {rnn_n_elems_last, { 0x0e, 2 }},
    {cnn_n_flt_iters, { 0x0e, 2 }},
    {pwl_n_segs, { 0x10, 2 }}, //  TODO:3: Activation Function implement ReLUHint and ZeroIndex
    {act_list_n_elems, { 0x12, 2 }},
    {cpy_n_elems, { 0x12, 2 }},
    {cnn_flt_size, { 0x12, 2 }},
    {bias_grp_cnt, { 0x12, 2 }},
    {cnn_n_flts_iter, { 0x14, 2 }},
    {bias_grp_value, { 0x14, 2 }},
    {cnn_n_flt_outs, { 0x16, 2 }},
    {cnn_flt_bf_sz_iter, { 0x18, 2 }},
    {cnn_flt_bf_sz_last, { 0x1A, 2 }},
    {in_buffer, { 0x20, 4 }},
    {gmm_descriptor, { 0x20, 4 }},
    {out_buffer, { 0x24, 4 }},
    {out_sum_buffer, { 0x28, 4 }},
    {rnn_out_fb_buffer, { 0x2C, 4 }},
    {weight_buffer, { 0x30, 4 }},
    {bias_buffer, { 0x34, 4 }},
    {act_list_buffer, { 0x38, 4 }},
    {bias_grp_buffer, { 0x38, 4 }},
    {pwl_seg_def_buffer, { 0x3c, 4 }},

    {cnn2d_in_dim_d, { 0x02, 2 }},
    {cnn2d_in_dim_w, { 0x04, 2 }},
    {cnn2d_in_dim_h, { 0x06, 2 }},
    {cnn2d_pool_stride_h, { 0x08, 1 }},
    {cnn2d_pool_stride_w, { 0x09, 1 }},
    {cnn2d_padding_h, { 0x0E, 1 }},
    {cnn2d_padding_w, { 0x0F, 1 }},
    {cnn2d_conv_stride_h, { 0x14, 1 }},
    {cnn2d_conv_stride_w, { 0x15, 1 }},
    {cnn2d_conv_out_w, { 0x16, 2 }},
    {cnn2d_conv_out_h, { 0x18, 2 }},
    {cnn2d_pool_out_w, { 0x1A, 2 }},
    {cnn2d_pool_out_h, { 0x1C, 2 }},
    {cnn2d_pool_window_w, { 0x1E, 1 }},
    {cnn2d_pool_window_h, { 0x1F, 1 }},
    {cnn2d_addaptive, { 0x28, 4 }},
    {cnn2d_kernel_scalar, { 0x30, 1 }},
    {cnn2d_conv_kernel_w, { 0x2C, 1 }},
    {cnn2d_conv_kernel_h, { 0x2D, 1 }},
    {cnn2d_kernel_iter, { 0x38, 1 }},
    {cnn2d_kernel_wg, { 0x3A, 2 }},
    {cnn2d_bias_mode, { 0x0b, 1, 3, 1,
        {
            { GNA_BIAS_PER_STRIDE, static_cast<uint8_t>(0) },
            { GNA_BIAS_PER_KERNEL, static_cast<uint8_t>(1) },
       }}},
};

const std::map<const XnnParameterType, const XnnParameter>& LayerDescriptor::getParameterMap(const gna_device_version hwId)
{

    static const std::map<const gna_device_version, const std::map<const XnnParameterType, const XnnParameter>&> parameterMap =
    {
        {GNA_CNL, XnnDescriptorGNA_1},
        {GNA_GLK, XnnDescriptorGNA_1},
        {GNA_ICL, XnnDescriptorGNA_1},
        {GNA_TGL, XnnDescriptorGNA_1},
        {GNA_ADL, XnnDescriptorGNA_3},
        {GNA_SUE_CREEK, XnnDescriptorGNA_1},
        {GNA_JELLYFISH, XnnDescriptorGNA_1},
        {GNA_ACE_EMBEDDED, XnnDescriptorGNA_3},
        {GNA_ACE_ANNA, XnnDescriptorGNA_3},
    };
    return parameterMap.at(hwId);
}

LayerDescriptor::LayerDescriptor(const BaseAddress memoryBaseIn, const BaseAddress& addressIn,
    const AccelerationDetector& detector) :
    LayerDescriptor{
        {},
        getSize(detector.GetDeviceVersion()),
        detector,
        memoryBaseIn,
        addressIn,
        getParameterMap(detector.GetDeviceVersion()) }
{
};

LayerDescriptor::LayerDescriptor(const LayerDescriptor& base, AddrGmmCfg gmmDescriptor) :
     LayerDescriptor{
        gmmDescriptor,
        base.Size,
        base.Detector,
        base.memoryBase,
        base.address,
        *base.xnnReferenceParams }
{
};

LayerDescriptor::LayerDescriptor(const AddrGmmCfg gmmConfig, const size_t size,
    const AccelerationDetector& detector, const BaseAddress memoryBaseIn, BaseAddress descriptorBaseIn,
    const std::map<const XnnParameterType, const XnnParameter>& paramsIn) :
    Size{ size },
    Detector{ detector },
    GmmDescriptor{ gmmConfig },
    memoryBase{ memoryBaseIn },
    address{ descriptorBaseIn },
    offset{ address.GetOffset(memoryBase) },
    xnnReferenceParams{ &paramsIn },
    gmmReferenceParams{ &GmmDescriptorGNA }
{
    Expect::ValidBuffer(address);
    Expect::AlignedTo(address, static_cast<uint32_t>(Size));
    if (GmmDescriptor)
    {
        Expect::ValidBuffer(GmmDescriptor);
        Expect::AlignedTo(GmmDescriptor, sizeof(GMM_CONFIG));
    }
}
