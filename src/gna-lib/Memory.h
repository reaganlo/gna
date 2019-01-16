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

#include <memory>
#include <map>

#include "common.h"
#include "Address.h"
#include "IoctlSender.h"

namespace GNA
{
    class AccelerationDetector;
    class CompiledModel;

    class Memory : public BaseAddress
    {
    public:
        Memory() = default;

        // just makes object from arguments
        Memory(void * buffer, const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount, IoctlSender &sender);

        // allocates and zeros memory
        Memory(const size_t userSize, const uint16_t layerCount, const uint16_t gmmCount, IoctlSender &sender);

        virtual ~Memory();

        void Map();

        uint64_t GetId() const;

        size_t GetSize() const
        {
            return size;
        }

        template<class T = void> T * GetUserBuffer() const
        {
            auto address = BaseAddress(this->Get() + InternalSize);
            return address.Get<T>();
        }

        void AllocateModel(const gna_model_id modelId, const gna_model * model, const AccelerationDetector& detector);

        void DeallocateModel(gna_model_id modelId);

        CompiledModel& GetModel(gna_model_id modelId) const;
        void * GetDescriptorsBase(gna_model_id modelId) const;

        std::map<gna_model_id, std::unique_ptr<CompiledModel>> Models;

        // Internal GNA library auxiliary memory size.
        const size_t InternalSize;

        // Size of memory requested for model by user.
        const size_t ModelSize;


    protected:
        virtual std::unique_ptr<CompiledModel> createModel(const gna_model_id modelId, const gna_model *model, const AccelerationDetector &detector);

        void unmap();

        uint64_t id = 0;

        size_t size = 0;

        size_t descriptorsSize = 0;

        IoctlSender &ioctlSender;

        bool mapped = false;

        std::map<gna_model_id, void*> modelDescriptors;
    };

}
