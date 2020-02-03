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

#pragma once

#include "TlvErrors.h"

#include <stdint.h>

#define TLV_TYPE_ID_SIZE 4
#define TLV_LENGTH_SIZE 4

typedef union TlvTypeIdImplementation
{
    char stringValue[TLV_TYPE_ID_SIZE];
    uint32_t numberValue;
} TlvTypeId;

    /**
    TlvFrame structure:
     ->type->TlvTypeId*
        -size of the type is provided by TLV_TYPE_SIZE constantaw
        -ends without '\0' at the end
     ->length->uint32_t:
        -describes number of bytes in data sect
     ->value->uint8_t*:
        -equals NULL for frames which hols children frames,
        -points to DATA for leaf TLV frames. See the description
            of leaf TLV frame at the begging of the file.
     ->parentNode->TlvFrame:
        -indicates parent frame
        -NULL is set for frames at the top of the hierarchy tree

     Leaf TLV frame - brings just information, can bring some data
     and does not consist of any other tlv records.
     ->numberOfChildrenNodes->uint32_t
     ->childrenNodes->TlvFrame:
        -pointer to the first children TlvFrame.
        -tlv frames are returned in sequence, one after the other
    */
struct TlvFrame
{
    TlvTypeId type;

    uint32_t length;

    const uint8_t* value;

    struct TlvFrame* parentNode;

    uint32_t numberOfChildrenNodes;

    struct TlvFrame* childrenNodes;
};

#ifdef __cplusplus
extern "C"
{
#endif
    /**
    It is REQUIRED to use this function once before creating any TlvFrames
    to write.

    Initializes id counters and nulls frame list pointer.
    It should be used only before the first attempt to create TlvFrame.
    */
    void TlvLibraryInit();

    /**
    Passes messasge of last error in string format.
    */
    void TlvGetStatusMessage(int from, const char** dest);

    /**
     Determines size needed to allocate memory for frames.
     @param [in] pointer to data which will be decoded
     @param [in] size of data provided by the user
     @param [out] calculated size needed for returning read TLV frames
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvGetSize(void* data, uint32_t inputSize, uint32_t* returnedSize);

    /**
     Reads Tlv frames from data provided by the user.

     @param [in] pointer to data which should be decoded
     @param [in] size of data provided by the user
     @param [out] pointer to data to which decoded frames will be written.
        User should provide proper size of a memory. Use TlvGetSize to\
        determine required size.
     @return TlvStatus - the status code returned from function
    */
    enum TlvStatus TlvDecode(void* data, uint32_t size, struct TlvFrame* memory,
        uint32_t* numbberOfreadFrames);

    /**
     Adds Tlv record/layer, to which later user is expected to add other
     record/layer with data or other complex structure. It informs about
     the hierarchy of the tlv frame or just simply carry information using
     header.

     @param [in] type name, should be the same length as TLV_TYPE_NAME_SIZE,
     @param [out] returns id to which current record was written
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvRecordInit(const TlvTypeId type, uint32_t* id);

    /**
     Adds Tlv record which is the smallest unit. See the description
     of leaf TLV frame at the begging of the file.

     @param [in] type name, should be the same length as TLV_TYPE_NAME_SIZE,
     @param [in] size of data, provided by the user to be assigned to the record
     @param [in] pointer to data to be assigned to the record
     @param [out] returns id to which current record was written
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvRecordInitRaw(const TlvTypeId type, uint32_t length,
        const void* value, uint32_t* id);

    /**
     Makes link between two records. It allows to assign lower records/layers
     to upper ones.

     @param [in] id of the parent record
     @param [in] id of the child reccord
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvRecordAdd(uint32_t parentRecordId, uint32_t childRecordId);

    /**
     Calculates size of the the pointed record and all records assigned to it
     @param [in] id of the record
     @param [out] calculated size
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvRecordGetSize(uint32_t id, uint32_t* recordSizeOut);

    /**
     Saves data prepared by the user to the pointed memory
     @param [in] id of the record on the highest layer
     @param [in] size of allocated memory
     @param [out] ] pointer to data to which ready tlv will be written.
        User should provide proper size of a memory. Use TlvRecordGetSize to\
        determine required size.
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvSerialize(uint32_t id, void* data, const uint32_t dataSize);

    /**
     Releases record as well as child records. This function
     will delte children only if the link between records exists.
     @param [in] id of the node to release, and its children Nodes
     @return TlvStatus - the status code returned from function
     */
    enum TlvStatus TlvRecordsRelease(uint32_t id);

    /**
     Loads list with own leaf TLV types.
     Leaf TLV type is a type of frame that only provides data,
     it cannot be the parent to another frame.
     Every type on the list should be terminated with '\0'
     @param [in] pointer to data /list with params
     @param [in] number of elemements on the list
     */
    enum TlvStatus TlvLoadOwnRawList(const TlvTypeId ownRawTypeList[],
        const uint32_t numberOfRawElements);


#ifdef __cplusplus
}
#endif
