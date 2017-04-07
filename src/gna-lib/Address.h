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

#include "common.h"

#include "GnaConfig.h"

namespace GNA
{

// Generic not defined Address class
template<typename T> class Address;

// Address class for operating on const pointers
template<typename T> class Address<T*const>
{
public:
    Address() = default;
    Address(void * const value) :
        buffer(static_cast<T*>(value))
    {}
    Address(const void * const value) :
        buffer(static_cast<T*>(const_cast<void*>(value)))
    {}
    Address(const Address& address) :
        buffer(address.buffer)
    {}
    template<class C> Address(const Address<C*>& address) :
        buffer(address.Get())
    {}
    template<class C> Address(const Address<C*const>& address) :
        buffer(address.Get())
    {}
    ~Address() = default;

    Address operator+(const uint32_t& right) const
    {
        Address plus(*this);
        plus.buffer = plus.Get<T>() + right;
        return plus;
    }
    Address operator-(const uint32_t& right) const
    {
        Address left(*this);
        left.buffer = left.Get<T>() - right;
        return left;
    }

    bool operator ==(const std::nullptr_t &right) const
    {
        return right == buffer;
    }

    explicit operator bool() const
    {
        return (nullptr != buffer);
    }

    bool operator!() const
    {
        return (nullptr == buffer);
    }

    template<class X> operator X* const() const
    {
        return static_cast<X * const>(buffer);
    }

    template<class X> X * const Get() const
    {
        return static_cast<X * const>(buffer);
    }

    T * const Get() const
    {
        return static_cast<T * const>(buffer);
    }

    T& operator*() const
    {
        return *(static_cast<T*>(buffer));
    }

    template<class X> uint32_t GetOffset(const Address<X*const>& base) const
    {
        if (!this) return 0;
        return PtrToUint((uint8_t*)(this->Get<uint8_t>() - base.Get<uint8_t>()));
    }

    //TODO:INTEGRATION:remove when all buffers switched to Address
    template<class X> static uint32_t GetOffset(const void * const address, const Address<X*const>& base)
    {
        if (nullptr == address) return 0;
        return PtrToUint((uint8_t*)address - base.Get<uint8_t>());
    }

protected:
    void * buffer = nullptr;
};

// Address class for operating on non-const pointers
template<typename T> class Address<T *> : public Address<T*const>
{
public:
    Address() = default;
    Address(void * value) :
        Address<T*const>(value)
    {}
    Address(const void * value) :
        Address<T*const>(value)
    {}
    template<class C> Address(const Address<C*>& address) :
        Address<T*const>(address)
    {}
    template<class C> Address(const Address<C*const>& address) :
        Address<T*const>(address.Get())
    {}

    const Address& operator++(int)
    {
        this->buffer = this->Get<T>() + sizeof(T);
        return *this;
    }
    const Address& operator-=(const uint32_t& right)
    {
        this->buffer = this->Get<T>() - right;
        return *this;
    }
    const Address& operator+=(const uint32_t& right)
    {
        this->buffer = this->Get<T>() + right;
        return *this;
    }
    const Address& operator =(const Address& right)
    {
        this->buffer = right.buffer;
        return *this;
    }
    const Address& operator =(const T& right)
    {
        *this->buffer = right;
        return *this;
    }
};

// Address Aliases

using BaseAddressC = Address<uint8_t * const>;
//using AddressU8C = Address<uint8_t * const>;
//using AddressU16C = Address<uint16_t * const>;
//using AddressU32C = Address<uint32_t * const>;
//using AddressU64C = Address<uint64_t * const>;

using BaseAddress = Address<uint8_t *>;
//using AddressU8 = Address<uint8_t *>;
//using AddressU16 = Address<uint16_t *>;
//using AddressU32 = Address<uint32_t *>;
//using AddressU64 = Address<uint64_t *>;
using AddrGmmCfg = Address<GMM_CONFIG *>;
using AddrXnnLyr = Address<XNN_LYR *>;
using AddrGmmCfgC = Address<GMM_CONFIG * const>;
//using AddrXnnLyrC = Address<XNN_LYR * const>;

using InOutBuffer = Address<uint8_t * const>;
using GmmInputBuffer = Address<uint8_t * const>; // Input Buffer for GMM layer
using XnnInputBuffer = Address<uint16_t * const>; // Input Buffer for Neural layer
using OutputBuffer = Address<uint16_t * const>; // Activated Output Buffer
using BareOutputBuffer = Address<uint32_t * const>;  // Non-Activated Output Buffer

}
