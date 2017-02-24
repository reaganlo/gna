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

#include "KernelMacros.h"
#include "common.h"

#define CNNFilter16         KERNEL(CNNFilter16)
#define CNNFilterPool16     KERNEL(CNNFilterPool16)
#define SumPoolingFunction  KERNEL(SumPoolingFunction)
#define MaxPoolingFunction  KERNEL(MaxPoolingFunction)
#define MaxPartialPoolingFunction KERNEL(MaxPartialPoolingFunction)
#define SumPartialPoolingFunction KERNEL(SumPartialPoolingFunction)

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif


/* Calculates CNNFilter16
 *
 * @FMR     number of input columns
 * @FM      number of feature maps
 * @FMC     number of feature map columns
 * @FN      number of filters
 * @FC      number of filters coefficients
 * @I       input vectors pointer(non-interleaved)
 * @F       filters
 * @B       biases
 * @O       output matrix
 * @nSat    number of saturations found
 */
void
CNNFilter16(
    const   uint32_t    IC,
    const   uint32_t    FM,
    const   uint32_t    FMC,
    const   uint32_t    FN,
    const   uint32_t    FC,
            int16_t*    I,
            int16_t*    F,
            nn_bias_s*  B,
            int32_t*    O,
            uint32_t*   nSat
);
   
/* Calculates CNNFilterPool16
 *
 * @IC      number of input columns
 * @FM      number of feature maps
 * @FMC     umber of feature map columns
 * @FN      number of filters
 * @FC      number of filter coeficients
 * @PS      number of pool size
 * @PSTEP   number of pool step
 * @NS      number of segments
 * @S       pointer to segments
 * @B       pointer to biases
 * @F       pointer to filters
 * @I       pointer to inputs
 * @O       pointer to outputs
 * @nSat    number of saturations found
 * @PT      pool type
 * @pwlBuff PWL unpacked buffer
 * @pool    Pool buffer
 */
void
CNNFilterPool16(
    const   uint32_t            IC,
    const   uint32_t            FM,
    const   uint32_t            FMC,
    const   uint32_t            FN,
    const   uint32_t            FC,
    const   uint32_t            PS,
    const   uint32_t            PSTEP,
    const   uint32_t            NS,
            nn_pwl_seg*         S,
            nn_bias_s*          B,
            int16_t*            F,
            int16_t*            I,
            int16_t*            O,
            uint32_t*           nSat,
            nn_pool_type        PT,
            void*               pwlBuff,
            int64_t*            pool
);


/* Calculates SumPoolingFunction
* @PS   number of pool size
* @P    pointer to pool array
* @V    pointer to value
*/
void
SumPoolingFunction(
    const   uint32_t    PS,
            int64_t*    P,
            int64_t*    V
);

/* Calculates MaxPoolingFunction
* @PS   number of pool size
* @P    pointer to pool array
* @V    pointer to value
*/

void
MaxPoolingFunction(
    const   uint32_t    PS,
            int64_t*    P,
            int64_t*    V
);

/* Calculates MaxPartialPoolingFunction
* @PS   number of pool size
* @PNE  number of pool entries
* @PSI  number of pool start index
* @P    pointer to pool array
* @V    pointer to value
*/
void
MaxPartialPoolingFunction(
    const   uint32_t    PS,
    const   int32_t     PNE,
    const   uint32_t    PSI,
            int64_t*    P,
            int64_t*    V
);


/* Calculates SumPartialPoolingFunction
* @PS   number of pool size
* @PNE  number of pool entries
* @PSI  number of pool start index
* @P    pointer to pool array
* @V    pointer to value
*/
void
SumPartialPoolingFunction(
    const   uint32_t    PS,
    const   int32_t     PNE,
    const   uint32_t    PSI,
            int64_t*    P,
            int64_t*    V
);

inline void
saturate64_store_out(
int64_t*            out,
uint32_t*           nSat)
{
    if (*out > INT32_MAX)
    {
        *out = INT32_MAX;
        (*nSat)++;
    }
    else if (*out < INT32_MIN)
    {
        *out = INT32_MIN;
        (*nSat)++;
    } 
}


#ifdef __cplusplus
}
#endif
