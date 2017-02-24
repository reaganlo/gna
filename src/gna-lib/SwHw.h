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

#include <map>
#include <future>

#include "GmmLayer.h"
#include "GnaConfig.h"
#include "SoftwareModel.h"
#include "RequestConfiguration.h"

namespace GNA
{
#ifdef _WIN32
typedef OVERLAPPED  io_handle_t;    // ioctl handle type for Windows
#else
typedef uint32_t    io_handle_t;    // ioctl handle type for Linux and android
#endif

/**
 * Software request handle
 */
class Sw
{
public:
    /**
     * Creates empty software request handle
     */
    Sw();

    virtual ~Sw() {};

    /**
     * software processing event handle
     */
    std::future<status_t> handle;

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    Sw(const Sw &) = delete;
    Sw& operator=(const Sw&) = delete;
};

/**
 * Hardware request handle
 */
class Hw : public Sw
{
public:
     /**
     * Creates empty hardware request handle
     *
     * @base    pointer to memory buffer start
     * @inBuffSize  Internal hw input buffer size in KB
     */
    Hw( const void* base,
        uint32_t    inBuffSize);

    /**
     * Destroys hardware request handle resources if any
     */
    virtual ~Hw();

    /**
     * hardware processing event handle
     */
    io_handle_t      io_handle;

   /**
    *  hardware request input data buffer, 
    *   NOTE: pointer used for whole data buffer including:
    *       - configuration data (hw_calc_in_t) at the beginning of the buffer 
    *       - xnn layer descriptor (varying size) data (in xNN mode) after hw_calc_in_t
    */
    hw_calc_in_t*   inData;

    // pointer to memory buffer for xnn layer descriptors
    XNN_LYR *xnnLayerDescriptors;

    // pointer to memory buffer for all GMM Layer Hardware descriptors
    GMM_CONFIG *gmmLayerDescriptors;

    // Total size of request data
    uint32_t dataSize;

    /**
     * Fills SoftwareModel configuration data based on request parameters
     * @model     SoftwareModel configuration
     */
    void Fill(SoftwareModel* model);

    /**
     * Gets integer offset of address from the beginning of memory buffer
     *
     * @address (in) pointer to data in allocated memory buffer
     * @buffer    (in) pointer to memory buffer start
     * @return  integer address offset
     */
    inline static uint32_t getAddrOffset(
        const void* address,
        const void* buffer)
    {
        if (nullptr == address) return 0;
        return PtrToUint((void*)((uint8_t*)address - (uint8_t*)buffer));
    }

protected:
    static const std::map<const gna_gmm_mode, const GMM_MODE_CTRL> GmmModes;
    /**
     * pointer to memory buffer start
     */
    const void*     base;

    /**
     * Internal hw input buffer size in KB
     */
    uint32_t        inBuffSize;

     /**
     * Gets integer offset of address from the beginning of memory buffer
     *
     * @address     (in)    pointer to data in allocated memory buffer
     * @return      integer address offset
     */
    inline uint32_t getAddrOffset(const void* address)
    {
        if (nullptr == address) return 0;
        return PtrToUint((void*)((uint8_t*)address - (uint8_t*)base));
    }

    /**
     * Initiates request
     */
    void init();

    void GmmLayerDescriptorSetup(const GmmLayer *layer, GMM_CONFIG* descriptor);
    void GmmLayerDescriptorUpdateRequestConfig(const uint32_t layerIndex, const GmmLayer *layer, const RequestConfiguration &configurationm, GMM_CONFIG* descriptor);
    void GmmLayerDescriptorUpdateInput(const ConfigurationBuffer &inputBuffer, GMM_CONFIG* descriptor);
    void GmmLayerDescriptorUpdateOutput(const ConfigurationBuffer &outputBuffer, GMM_CONFIG* descriptor);
    void GmmLayerDescriptorUpdateActiveList(const GmmLayer *gmm, const ActiveList &activeList, GMM_CONFIG* descriptor);

    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    Hw(const Hw &) = delete;
    Hw& operator=(const Hw&) = delete;
};

}
