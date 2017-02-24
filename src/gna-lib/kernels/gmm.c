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

#include "kernel-gmm.h"

#define gmm_maxmix_8u8u_32u     KERNEL(gmm_maxmix_8u8u_32u)
#define gmm_maxmix_8u16u_32u    KERNEL(gmm_maxmix_8u16u_32u)
#define gmm_maxmix_8u8u_32u_g1  KERNEL(gmm_maxmix_8u8u_32u_g1)
#define gmm_maxmix_8u8u_32u_g2  KERNEL(gmm_maxmix_8u8u_32u_g2)
#define gmm_maxmix_8u8u_32u_g3  KERNEL(gmm_maxmix_8u8u_32u_g3)
#define gmm_maxmix_8u8u_32u_g4  KERNEL(gmm_maxmix_8u8u_32u_g4)
#define gmm_maxmix_8u8u_32u_g5  KERNEL(gmm_maxmix_8u8u_32u_g5)
#define gmm_maxmix_8u8u_32u_g6  KERNEL(gmm_maxmix_8u8u_32u_g6)
#define gmm_maxmix_8u8u_32u_g7  KERNEL(gmm_maxmix_8u8u_32u_g7)
#define gmm_maxmix_8u8u_32u_g8  KERNEL(gmm_maxmix_8u8u_32u_g8)

#pragma warning (disable : 592 )

// Rather combining mixtures using log addition, this function selects
// the score of the single best scoring mixture as the representative.
// In unsigned version, score is negative of true score and best score
// is the smallest score.

#if OPT_LEVEL == 0 || OPT_LEVEL == 1 // NONE

uint32_t
gmm_maxmix_8u8u_32u(
	_GMM8_ARGS)
{
  const uint8_t *pM = pMeans;
  const uint8_t *pV = pVars;
  const uint32_t *pC = pGconst;
  uint32_t MinScore32u = ScoreLimit32u;
  uint64_t Sum64u;
  uint32_t i,j;
  
  for(i = 0; i < nMixtures; i++)
  {
    uint32_t Score32u = 0;

    for(j = 0; j < nVecElements; j++)
    {
      int16_t Diff16s = (int16_t)pFeat[j] - (int16_t)pM[j];
      uint16_t SqrDiff16s = (uint16_t)(Diff16s * Diff16s);

      Score32u += (uint32_t)SqrDiff16s * (uint32_t)pV[j];
    }

    // sum may saturate depending on value of const
    Sum64u = (uint64_t) Score32u + (uint64_t) *pC;
    Score32u = (Sum64u > 0xffffffff) ? 0xffffffff : (uint32_t) Sum64u;
    
    MinScore32u = (Score32u < MinScore32u) ? Score32u : MinScore32u;
  
    pM += nVecElements;
    pV += nVecElements;
    pC++;
  }

  return(MinScore32u);
}

uint32_t
gmm_maxmix_8u16u_32u(
	_GMM16_ARGS)
{
  const uint8_t *pM = pMeans;
  const uint16_t *pV = pVars;
  const uint32_t *pC = pGconst;
  uint32_t MinScore32u = ScoreLimit32u;
  uint64_t Sum64u;
  uint32_t i,j;
  
  for(i = 0; i < nMixtures; i++)
  {
    uint64_t Score64u = 0;
    uint32_t Score32u;

    for(j = 0; j < nVecElements; j++)
    {
      int16_t Diff16s = (int16_t)pFeat[j] - (int16_t)pM[j];
      uint16_t SqrDiff16s = (uint16_t)(Diff16s * Diff16s);

      Score64u += (uint64_t)SqrDiff16s * (uint64_t)pV[j];
    }

    // sum may saturate depending on value of const
    Sum64u = Score64u + (uint64_t) *pC;
    Score32u = (Sum64u > 0xffffffff) ? 0xffffffff : (uint32_t) Sum64u;
    
    MinScore32u = (Score32u < MinScore32u) ? Score32u : MinScore32u;
  
    pM += nVecElements;
    pV += nVecElements;
    pC++;
  }
  
  return(MinScore32u);
}

#endif //#if OPT_LEVEL == 0 || OPT_LEVEL == 1 // NONE

#if OPT_LEVEL > 1 // SSE4+

#include "immintrin.h"

uint32_t
gmm_maxmix_8u8u_32u(
	_GMM8_ARGS)
{
  const uint8_t *pM = pMeans;
  const uint8_t *pV = pVars;
  const uint32_t *pC = pGconst;
  uint64_t MinScore64u = ScoreLimit32u;
  uint64_t Sum64u;
  uint32_t i,j;
  
  for(i = 0; i < nMixtures; i++)
  {
	__m128i sum;
    __m128i sum_1 = _mm_xor_si128(sum_1, sum_1);
	__m128i sum_2 = sum_1;

    for(j = 0; j < nVecElements; j+=8)
    {
        __m128i load1 = CVT64_128(&pFeat[j]); // vector load 8x8-bit
        __m128i load2 = CVT64_128(&pM[j]); // vector load 8x8-bit
        __m128i load3 = CVT64_128(&pV[j]); // vector load 8x8-bit
        __m128i zext1 = _mm_cvtepu8_epi16(load1); // convert to 8x16-bit
        __m128i zext2 = _mm_cvtepu8_epi16(load2); // convert to 8x16-bit
        __m128i zext3 = _mm_cvtepu8_epi16(load3); // convert to 8x16-bit
        __m128i diff16s = _mm_sub_epi16(zext1, zext2); // 8x16-bit subtract
        __m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
        __m128i prod_low = _mm_mullo_epi16(sqrdiff16s, zext3); // 8x16-bit mult (lo part)
        __m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, zext3); // 8x16-bit mul (hi part)
        __m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
        __m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
        sum_1 = _mm_add_epi32(sum_1, lower_prods); // 4x32-bit addition
        sum_2 = _mm_add_epi32(sum_2, upper_prods); // 4x32-bit addition
    }
	sum = _mm_add_epi32(sum_1, sum_2); // 4x32-bit addition
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xee)); // horizontal 32-bit add
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x55)); // horizontal 32-bit add

    // sum may saturate depending on value of const
    Sum64u = (uint64_t) _mm_cvtsi128_si32(sum) + (uint64_t) *pC;
    MinScore64u = (Sum64u < MinScore64u) ? Sum64u : MinScore64u;
    
    pM += nVecElements;
    pV += nVecElements;
    pC++;
  }
  
  return(MinScore64u);
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g1_sse4
void
gmm_maxmix_8u8u_32u_g1(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j,k;
	uint64_t MinScore64u; 
	uint64_t Sum64u, Sum64u2;
	MinScore64u = ScoreLimit32u;

	if ((nMixtures & 1) == 1)
	{
		__m128i sum_1         = _mm_setzero_si128();

		__m128i load1                = CVT64_128((__m128i*)pF);                                
		__m128i load2                = CVT64_128((__m128i*)pM); // vector load 8x8-bit                  
		__m128i load3                = CVT64_128((__m128i*)pV); // vector load 8x8-bit                                          

		for(j = 8; j <= nVecElements; j+=8)
		{	
			__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												
		
			__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
			__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				        load1        = CVT64_128((__m128i*)&pF[j]);                                
				        load2        = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit                  
				        load3        = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit                                          
			__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
			__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
			__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
			__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd

			sum_1 				= _mm_add_epi32(sum_1, lower_prods); 
			sum_1 				= _mm_add_epi32(sum_1, upper_prods); 
		}
		sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0xee)); // horizontal 32-bit add
		sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0x55)); // horizontal 32-bit add
		
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_1) + (uint64_t)*pC;
		MinScore64u     = Sum64u < MinScore64u ? Sum64u : MinScore64u;
		
		pM += nVecElements;
		pV += nVecElements;
		pC++;
		nMixtures--;
	}

	for(i = 0; i < nMixtures; i+=2)
	{
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();

		__m128i load1   = CVT64_128((__m128i*)pF);                                
		__m128i load2   = CVT64_128((__m128i*)pM); // vector load 8x8-bit                  
		__m128i load3   = CVT64_128((__m128i*)pV); // vector load 8x8-bit                                          
		__m128i load22  = CVT64_128((__m128i*)(pM+nVecElements)); // vector load 8x8-bit                 
		__m128i load33  = CVT64_128((__m128i*)(pV+nVecElements)); // vector load 8x8-bit                                         

		for(j = 8, k = 8 + nVecElements; j <= nVecElements; j+=8, k+=8)
		{     	
				__m128i load1_1      = _mm_cvtepu8_epi16(load1); 
				__m128i load2_1      = _mm_cvtepu8_epi16(load2);                 
				__m128i load3_1      = _mm_cvtepu8_epi16(load3);                                                                       

				__m128i diff16s      = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit            
				__m128i sqrdiff16s   = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)             
				        load1        = CVT64_128((__m128i*)&pF[j]);                                
				        load2        = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit                  
				        load3        = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit                                          

				__m128i prod_low     = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high    = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i load22_1     = _mm_cvtepu8_epi16(load22);               
				__m128i load33_1     = _mm_cvtepu8_epi16(load33);                                                                     

				__m128i lower_prods  = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods  = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd

				__m128i diff16s2     = _mm_sub_epi16(load1_1, load22_1); // convert to 8x16-bit            
				__m128i sqrdiff16s2  = _mm_mullo_epi16(diff16s2, diff16s2); // 8x16-bit mul (hi zero)               
					    load22       = CVT64_128((__m128i*)&pM[k]); // vector load 8x8-bit                 
		                load33       = CVT64_128((__m128i*)&pV[k]); // vector load 8x8-bit                                         

				__m128i prod_low2    = _mm_mullo_epi16(sqrdiff16s2, load33_1); // 8x16-bit mult (lo part)
				__m128i prod_high2   = _mm_mulhi_epu16(sqrdiff16s2, load33_1); // 8x16-bit mul (hi part)
				        sum_1        = _mm_add_epi32(sum_1, lower_prods); 
						sum_1        = _mm_add_epi32(sum_1, upper_prods); 

				__m128i lower_prods2 = _mm_unpacklo_epi16(prod_low2, prod_high2); // lo 4x32-bit prd
				__m128i upper_prods2 = _mm_unpackhi_epi16(prod_low2, prod_high2); // hi 4x32-bit prd

				        sum_2        = _mm_add_epi32(sum_2, lower_prods2); 
				        sum_2        = _mm_add_epi32(sum_2, upper_prods2); 
}

		__m128i horiz1 = _mm_hadd_epi32(sum_1, sum_2);
		horiz1 = _mm_hadd_epi32(horiz1, horiz1);
            
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(horiz1) + (uint64_t)*pC;
		MinScore64u     = Sum64u < MinScore64u ? Sum64u : MinScore64u;

		Sum64u2          = (uint64_t)_mm_extract_epi32(horiz1, 1) + (uint64_t)*(pC+1);
		MinScore64u      = Sum64u2 < MinScore64u ? Sum64u2 : MinScore64u;

		pM += (nVecElements<<1);
		pV += (nVecElements<<1);
		pC += 2;
	}

	*pScores = MinScore64u;
}

#endif //#if OPT_LEVEL > 1 // SSE4+

#if (OPT_LEVEL > 1) && (OPT_LEVEL < 6) // SSE4/AVX1 only (same code, different compile options)

uint32_t
gmm_maxmix_8u16u_32u(
	_GMM16_ARGS)
{
	const uint8_t  *pM = pMeans;
	const uint16_t *pV = pVars;
	const uint32_t *pC = pGconst;
	uint64_t MinScore64u = ScoreLimit32u;
	uint32_t i,j;
  
	__m128i zero = _mm_setzero_si128();

	for (i = 0; i < nMixtures; i++)
	{		
		uint64_t Score64u;
			
		__m128i sum_lo = zero;
		__m128i sum_hi = zero;
		__m128i load1 = CVT64_128(pFeat); // vector load 8x8-bit
		__m128i load2 = CVT64_128(pM); // vector load 8x8-bit

		for (j = 8; j <= nVecElements; j+=8)
		{
			__m128i zext1 = _mm_cvtepu8_epi16(load1); // convert to 8x16-bit
			__m128i zext2 = _mm_cvtepu8_epi16(load2); // convert to 8x16-bit
				
			__m128i load3 = _mm_loadu_si128((const __m128i*)pV); // vector load 8x8-bit

			__m128i diff16s = _mm_sub_epi16(zext1, zext2); // 8x16-bit subtract
			__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)
				
			pV += 8; 
			pM += 8;

			__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3); // 8x16-bit mult (lo part)
			__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3); // 8x16-bit mul (hi part)
				
			load1 = CVT64_128(&pFeat[j]); // vector load 8x8-bit
			load2 = CVT64_128(pM); // vector load 8x8-bit

			__m128i lower_prods = _mm_unpacklo_epi16(prod_low, zero); // lo 4x32-bit prd
			__m128i upper_prods = _mm_unpackhi_epi16(prod_low, zero); // hi 4x32-bit prd

			sum_lo = _mm_add_epi32(sum_lo, lower_prods); // 4x32-bit addition
			sum_lo = _mm_add_epi32(sum_lo, upper_prods); // 4x32-bit addition
			sum_hi = _mm_adds_epu16(sum_hi, prod_high);
		}
			
		__m128i sum_hi_2 = _mm_shuffle_epi32(sum_hi, 0xee);
		__m128i sum_hi_1 = _mm_cvtepu16_epi32(sum_hi);
		__m128i sum_hi_3 = _mm_cvtepu16_epi32(sum_hi_2);
			
		sum_hi = _mm_add_epi32(sum_hi_1, sum_hi_3);
		sum_lo = _mm_add_epi32(sum_lo, _mm_shuffle_epi32(sum_lo, 0xee)); // horizontal 32-bit add
		sum_hi = _mm_add_epi32(sum_hi, _mm_shuffle_epi32(sum_hi, 0xee)); // horizontal 32-bit add
		sum_lo = _mm_add_epi32(sum_lo, _mm_shuffle_epi32(sum_lo, 0x55)); // horizontal 32-bit add
		sum_hi   = _mm_add_epi32(sum_hi, _mm_shuffle_epi32(sum_hi, 0x55)); // horizontal 32-bit add
                  
		Score64u = _mm_cvtsi128_si32(sum_hi); // convert sum to 1x32-bit
		Score64u = (Score64u << 16) + _mm_cvtsi128_si32(sum_lo) + *pC; // convert sum to 1x32-bit

		// sum may saturate depending on value of const
    
		MinScore64u = (Score64u < MinScore64u) ? Score64u : MinScore64u;
    
		pC++;
	}

	return (MinScore64u);
}   

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g2_sse4 
void
gmm_maxmix_8u8u_32u_g2(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t  GConst64u;
	uint64_t  Scores64u[2];
	// init min scores
	uint64_t  MinScore = ScoreLimit32u;
	__m128i MinScores = CVT64_128((__m128i*) &MinScore); 	
	        MinScores = _mm_shuffle_epi32(MinScores, _MM_SHUFFLE(1,0,1,0));
		
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;

		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i load1         = _mm_load_si128((__m128i*)pF); 
		__m128i load2         = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3         = CVT64_128((__m128i*)pV); // vector load 8x8-bit						

		for(j = 8; j <= nVecElements; j+=8)
		{			
			pF += 16;
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												

			__m128i load1_2       = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
			__m128i load1_1       = _mm_cvtepu8_epi16(load1); 
					load1_2 = _mm_cvtepu8_epi16(load1_2);		
		
			__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
			__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
			__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
			__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		
			__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
			__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
			__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
			__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
			        load1         = _mm_load_si128((__m128i*)pF); 					
			__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
			__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
			__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd

			sum_1 				= _mm_add_epi32(sum_1, lower_prods); 
			sum_2 				= _mm_add_epi32(sum_2, lower_prods_2); 
			sum_1 				= _mm_add_epi32(sum_1, upper_prods); 
			sum_2 				= _mm_add_epi32(sum_2, upper_prods_2); 
		}
		GConst64u = *pC; 
		__m128i horiz1 = _mm_hadd_epi32(sum_1, sum_2);
		horiz1 = _mm_hadd_epi32(horiz1, horiz1);
		
		__m128i gconst  = CVT64_128((__m128i*)&GConst64u);
		        gconst  = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1,0,1,0));
		__m128i sum_64  = _mm_cvtepu32_epi64(horiz1);
				sum_64  = _mm_add_epi64(sum_64, gconst);
		__m128i cmpres  = _mm_cmpgt_epi64(MinScores, sum_64); 
		__m128i res_min = _mm_andnot_si128(cmpres, MinScores); 
		__m128i res_sum = _mm_and_si128(cmpres, sum_64); 
		MinScores = _mm_or_si128(res_min, res_sum);
		
		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}

	_mm_storeu_si128((__m128i*)Scores64u, MinScores);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g3_sse4
void
gmm_maxmix_8u8u_32u_g3(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t Sum64u;

	uint64_t  GConst64u;
	uint64_t  Scores64u[2];	
	uint64_t  MinScores_2 = ScoreLimit32u;
	__m128i MinScores_1 = CVT64_128((__m128i*) &MinScores_2); 	
	        MinScores_1 = _mm_shuffle_epi32(MinScores_1, _MM_SHUFFLE(1,0,1,0));
	
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i sum_3         = _mm_setzero_si128();		
		__m128i load1_1 = _mm_loadu_si128((__m128i*)pF); 
		__m128i load1_3 = CVT64_128((__m128i*)(pF+16)); 			
		__m128i load2   = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3   = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		
		for(j = 8; j <= nVecElements; j+=8)
		{			
			pF += 24;
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												

			__m128i load1_2 = _mm_shuffle_epi32(load1_1, _MM_SHUFFLE(3,2,3,2));
					load1_1 = _mm_cvtepu8_epi16(load1_1); 
					load1_2 = _mm_cvtepu8_epi16(load1_2);		
					load1_3 = _mm_cvtepu8_epi16(load1_3);		
					
			__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
			__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
			__m128i diff16s_3 = _mm_sub_epi16(load1_3, load2_1); // convert to 8x16-bit		
			__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
			__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
			__m128i sqrdiff16s_3 = _mm_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)		
			        load1_1 = _mm_loadu_si128((__m128i*)pF); 
			        load1_3 = CVT64_128((__m128i*)(pF+16)); 			
		            load2   = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
		            load3   = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit						
			__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
			__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
			__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
			__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
			__m128i prod_low_3 = _mm_mullo_epi16(sqrdiff16s_3, load3_1); // 8x16-bit mult (lo part)
			__m128i prod_high_3 = _mm_mulhi_epu16(sqrdiff16s_3, load3_1); // 8x16-bit mul (hi part)		
			
			__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
			__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
			__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
			__m128i lower_prods_3 = _mm_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
			__m128i upper_prods_3 = _mm_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd

			sum_1 				= _mm_add_epi32(sum_1, lower_prods); 
			sum_2 				= _mm_add_epi32(sum_2, lower_prods_2); 
			sum_3 				= _mm_add_epi32(sum_3, lower_prods_3); 
			sum_1 				= _mm_add_epi32(sum_1, upper_prods); 
			sum_2 				= _mm_add_epi32(sum_2, upper_prods_2); 
			sum_3 				= _mm_add_epi32(sum_3, upper_prods_3); 
		}
		sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0xee)); // horizontal 32-bit add			
		sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0x55)); // horizontal 32-bit add
		
				GConst64u    = *pC; 
				sum_1       = _mm_hadd_epi32(sum_1, sum_2);
		__m128i gconst      = CVT64_128((__m128i*)&GConst64u);
				sum_1       = _mm_hadd_epi32(sum_1, sum_1);
		        gconst      = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1,0,1,0));
				sum_1       = _mm_cvtepu32_epi64(sum_1);
				sum_1       = _mm_add_epi64(sum_1, gconst);
		__m128i cmpres_1    = _mm_cmpgt_epi64(MinScores_1, sum_1); 
		__m128i res_min_1   = _mm_andnot_si128(cmpres_1, MinScores_1); 
		__m128i res_sum_1   = _mm_and_si128(cmpres_1, sum_1); 
				MinScores_1 = _mm_or_si128(res_min_1, res_sum_1);

		        Sum64u      = (uint64_t)_mm_cvtsi128_si32(sum_3) + (uint64_t)*pC;
				MinScores_2 = Sum64u < MinScores_2 ? Sum64u : MinScores_2;

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	// store results
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[1];
	pScores[2]      = MinScores_2;
}

  //gmm_maxmix_8u8u_32u_grouped_opt_f8_g4_sse4
void
gmm_maxmix_8u8u_32u_g4(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t  GConst64;
	uint64_t  Scores64u[2];
	uint64_t  MinScore = ScoreLimit32u;
	__m128i MinScores_1 = CVT64_128((__m128i*) &MinScore); 	
	        MinScores_1 = _mm_shuffle_epi32(MinScores_1, _MM_SHUFFLE(1,0,1,0));
	__m128i MinScores_2 = MinScores_1;
	
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i sum_3         = _mm_setzero_si128();
		__m128i sum_4         = _mm_setzero_si128();		
			
		__m128i lower_prods;		
		__m128i upper_prods;		
		__m128i lower_prods_2;		
		__m128i upper_prods_2;		

		__m128i load1   = _mm_load_si128((__m128i*)pF); 
		__m128i load2   = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3   = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
			
		for(j = 8; j <= nVecElements; j+=8)
		{			
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												
			pF += 16;									
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
				        load2   = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
				        load3   = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit						
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1   = _mm_load_si128((__m128i*)pF); 
				        lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				        lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd				
			}
			pF += 16;									
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
				        sum_1 = _mm_add_epi32(sum_1, lower_prods);
				        sum_2 = _mm_add_epi32(sum_2, lower_prods_2);				
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        sum_1 = _mm_add_epi32(sum_1, upper_prods); 
				        sum_2 = _mm_add_epi32(sum_2, upper_prods_2); 

				        lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				        lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_3 = _mm_add_epi32(sum_3, lower_prods);
				sum_4 = _mm_add_epi32(sum_4, lower_prods_2);				
				sum_3 = _mm_add_epi32(sum_3, upper_prods); 
				sum_4 = _mm_add_epi32(sum_4, upper_prods_2); 
						load1   = _mm_load_si128((__m128i*)pF);
			}							
		}


				GConst64    = *pC; 
				sum_1       = _mm_hadd_epi32(sum_1, sum_2);
				sum_2       = _mm_hadd_epi32(sum_3, sum_4);
		__m128i gconst      = CVT64_128((__m128i*)&GConst64);
				sum_1       = _mm_hadd_epi32(sum_1, sum_1);
				sum_2       = _mm_hadd_epi32(sum_2, sum_2);
				gconst      = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1,0,1,0));
				sum_1       = _mm_cvtepu32_epi64(sum_1);
				sum_2       = _mm_cvtepu32_epi64(sum_2);
				sum_1       = _mm_add_epi64(sum_1, gconst);
				sum_2       = _mm_add_epi64(sum_2, gconst);
		__m128i cmpres_1    = _mm_cmpgt_epi64(MinScores_1, sum_1); 
		__m128i cmpres_2    = _mm_cmpgt_epi64(MinScores_2, sum_2); 
		__m128i res_min_1   = _mm_andnot_si128(cmpres_1, MinScores_1); 
		__m128i res_min_2   = _mm_andnot_si128(cmpres_2, MinScores_2); 
		__m128i res_sum_1   = _mm_and_si128(cmpres_1, sum_1); 
		__m128i res_sum_2   = _mm_and_si128(cmpres_2, sum_2); 
				MinScores_1 = _mm_or_si128(res_min_1, res_sum_1);
				MinScores_2 = _mm_or_si128(res_min_2, res_sum_2);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	// store results
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[1];
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_2);
	pScores[2]      = Scores64u[0];
	pScores[3]      = Scores64u[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g5_sse4
void
gmm_maxmix_8u8u_32u_g5(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t Sum64u;
	uint64_t  GConst64u;
	uint64_t  Scores64u[2];
	uint64_t  MinScores_3 = ScoreLimit32u;
	__m128i MinScores_1 = CVT64_128((__m128i*) &MinScores_3); 	
	        MinScores_1 = _mm_shuffle_epi32(MinScores_1, _MM_SHUFFLE(1,0,1,0));
	__m128i MinScores_2 = MinScores_1;

  
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i sum_3         = _mm_setzero_si128();
		__m128i sum_4         = _mm_setzero_si128();		
		__m128i sum_5         = _mm_setzero_si128();			
		__m128i load1         = _mm_loadu_si128((__m128i*)pF); 
		__m128i load2         = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3         = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		__m128i load1_3;			  
		for(j = 8; j <= nVecElements; j+=8)
		{			
			pF += 16;
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i	load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
			            load2   = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
			            load3   = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit						
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1    = _mm_loadu_si128((__m128i*)pF); 
				        load1_3  = CVT64_128((__m128i*)(pF+16)); 			
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_1 = _mm_add_epi32(sum_1, lower_prods);
				sum_2 = _mm_add_epi32(sum_2, lower_prods_2);				
				sum_1 = _mm_add_epi32(sum_1, upper_prods); 
				sum_2 = _mm_add_epi32(sum_2, upper_prods_2); 
			}
			pF +=24;
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
						load1_3 = _mm_cvtepu8_epi16(load1_3);		
						
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i diff16s_3 = _mm_sub_epi16(load1_3, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
				__m128i sqrdiff16s_3 = _mm_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)		
				        load1        = _mm_loadu_si128((__m128i*)pF); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				__m128i prod_low_3 = _mm_mullo_epi16(sqrdiff16s_3, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_3 = _mm_mulhi_epu16(sqrdiff16s_3, load3_1); // 8x16-bit mul (hi part)		
				
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				__m128i lower_prods_3 = _mm_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
				__m128i upper_prods_3 = _mm_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd

				sum_3 				= _mm_add_epi32(sum_3, lower_prods); 
				sum_4 				= _mm_add_epi32(sum_4, lower_prods_2); 
				sum_5 				= _mm_add_epi32(sum_5, lower_prods_3); 
				sum_3 				= _mm_add_epi32(sum_3, upper_prods); 
				sum_4 				= _mm_add_epi32(sum_4, upper_prods_2); 
				sum_5 				= _mm_add_epi32(sum_5, upper_prods_3); 
			}							
		}
				GConst64u   = *pC; 
				sum_1       = _mm_hadd_epi32(sum_1, sum_2);
				sum_2       = _mm_hadd_epi32(sum_3, sum_4);
		__m128i gconst      = CVT64_128((__m128i*)&GConst64u);
				sum_1       = _mm_hadd_epi32(sum_1, sum_1);
				sum_2       = _mm_hadd_epi32(sum_2, sum_2);
		        gconst      = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1,0,1,0));
				sum_1       = _mm_cvtepu32_epi64(sum_1);
				sum_2       = _mm_cvtepu32_epi64(sum_2);
				sum_1       = _mm_add_epi64(sum_1, gconst);
				sum_2       = _mm_add_epi64(sum_2, gconst);
		__m128i cmpres_1    = _mm_cmpgt_epi64(MinScores_1, sum_1); 
		__m128i cmpres_2    = _mm_cmpgt_epi64(MinScores_2, sum_2); 
		__m128i res_min_1   = _mm_andnot_si128(cmpres_1, MinScores_1); 
		__m128i res_min_2   = _mm_andnot_si128(cmpres_2, MinScores_2); 
		__m128i res_sum_1   = _mm_and_si128(cmpres_1, sum_1); 
		__m128i res_sum_2   = _mm_and_si128(cmpres_2, sum_2); 
				MinScores_1 = _mm_or_si128(res_min_1, res_sum_1);
				MinScores_2 = _mm_or_si128(res_min_2, res_sum_2);
		sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0xee)); // horizontal 32-bit add					
		sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0x55)); // horizontal 32-bit add

		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_5) + (uint64_t)*pC;
		MinScores_3     = Sum64u < MinScores_3 ? Sum64u : MinScores_3;

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[1];
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_2);
	pScores[2]      = Scores64u[0];
	pScores[3]      = Scores64u[1];
	pScores[4]      = MinScores_3;
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g6_sse4
void
gmm_maxmix_8u8u_32u_g6(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t Sum64u;
	uint64_t MinScores[6];

	MinScores[0] = ScoreLimit32u;
	MinScores[1] = ScoreLimit32u;
	MinScores[2] = ScoreLimit32u;
	MinScores[3] = ScoreLimit32u;    
	MinScores[4] = ScoreLimit32u;
	MinScores[5] = ScoreLimit32u;
	
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i sum_3         = _mm_setzero_si128();
		__m128i sum_4         = _mm_setzero_si128();		
		__m128i sum_5         = _mm_setzero_si128();		
		__m128i sum_6         = _mm_setzero_si128();		
		
		__m128i upper_prods;		
		__m128i upper_prods_2;		
		__m128i load1         = _mm_load_si128((__m128i*)pF); 
		__m128i load2         = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3         = CVT64_128((__m128i*)pV); // vector load 8x8-bit						

		for(j = 8; j <= nVecElements; j+=8)
		{			
			pF += 16;
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
		                load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
		                load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit						
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1         = _mm_load_si128((__m128i*)pF); 
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods   = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_1 = _mm_add_epi32(sum_1, lower_prods);
				sum_2 = _mm_add_epi32(sum_2, lower_prods_2);				
			}
			pF += 16;									
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i	load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
   					    sum_1 = _mm_add_epi32(sum_1, upper_prods); 
				        sum_2 = _mm_add_epi32(sum_2, upper_prods_2); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1 = _mm_load_si128((__m128i*)pF); 		
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_3 = _mm_add_epi32(sum_3, lower_prods);
				sum_4 = _mm_add_epi32(sum_4, lower_prods_2);				
			}							
			pF += 16;	
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
				        sum_3 = _mm_add_epi32(sum_3, upper_prods); 
				        sum_4 = _mm_add_epi32(sum_4, upper_prods_2); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1 = _mm_load_si128((__m128i*)pF); 
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_5 = _mm_add_epi32(sum_5, lower_prods);
				sum_6 = _mm_add_epi32(sum_6, lower_prods_2);				
				sum_5 = _mm_add_epi32(sum_5, upper_prods); 
				sum_6 = _mm_add_epi32(sum_6, upper_prods_2); 
			}										
		}
		sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0xee)); // horizontal 32-bit add
		sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0xee)); // horizontal 32-bit add			
		sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0x55)); // horizontal 32-bit add
		sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0x55)); // horizontal 32-bit add
		sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0xee)); // horizontal 32-bit add
		sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0xee)); // horizontal 32-bit add			
		sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0x55)); // horizontal 32-bit add
		sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0x55)); // horizontal 32-bit add
		sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0xee)); // horizontal 32-bit add
		sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0xee)); // horizontal 32-bit add		
		sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0x55)); // horizontal 32-bit add
		sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0x55)); // horizontal 32-bit add


		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_1) + (uint64_t)*pC;
		MinScores[0]    = Sum64u < MinScores[0] ? Sum64u : MinScores[0];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_2) + (uint64_t)*pC;
		MinScores[1]    = Sum64u < MinScores[1] ? Sum64u : MinScores[1];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_3) + (uint64_t)*pC;
		MinScores[2]    = Sum64u < MinScores[2] ? Sum64u : MinScores[2];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_4) + (uint64_t)*pC;
		MinScores[3]    = Sum64u < MinScores[3] ? Sum64u : MinScores[3];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_5) + (uint64_t)*pC;
		MinScores[4]    = Sum64u < MinScores[4] ? Sum64u : MinScores[4];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_6) + (uint64_t)*pC;
		MinScores[5]    = Sum64u < MinScores[5] ? Sum64u : MinScores[5];

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	pScores[0]      = MinScores[0];
	pScores[1]      = MinScores[1];
	pScores[2]      = MinScores[2];
	pScores[3]      = MinScores[3];
	pScores[4]      = MinScores[4];
	pScores[5]      = MinScores[5];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g7_sse4
void
gmm_maxmix_8u8u_32u_g7(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t Sum64u;
	uint64_t MinScores[8];

	MinScores[0] = ScoreLimit32u;
	MinScores[1] = ScoreLimit32u;
	MinScores[2] = ScoreLimit32u;
	MinScores[3] = ScoreLimit32u;    
	MinScores[4] = ScoreLimit32u;
	MinScores[5] = ScoreLimit32u;
	MinScores[6] = ScoreLimit32u;
		
	  
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i sum_3         = _mm_setzero_si128();
		__m128i sum_4         = _mm_setzero_si128();		
		__m128i sum_5         = _mm_setzero_si128();			
		__m128i sum_6         = _mm_setzero_si128();			
		__m128i sum_7         = _mm_setzero_si128();			
		__m128i upper_prods;		
		__m128i upper_prods_2;		
		__m128i load1         = _mm_loadu_si128((__m128i*)pF); 
		__m128i load1_3; 
		__m128i load2         = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3         = CVT64_128((__m128i*)pV); // vector load 8x8-bit						

		for(j = 8; j <= nVecElements; j+=8)
		{			
			pF += 16;
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
		                load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
		                load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit						
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1         = _mm_loadu_si128((__m128i*)pF); 
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods   = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_1 = _mm_add_epi32(sum_1, lower_prods);
				sum_2 = _mm_add_epi32(sum_2, lower_prods_2);				
			}
			pF += 16;									
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i	load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
   					    sum_1 = _mm_add_epi32(sum_1, upper_prods); 
				        sum_2 = _mm_add_epi32(sum_2, upper_prods_2); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
						load1 = _mm_loadu_si128((__m128i*)pF); 
						load1_3 = CVT64_128((__m128i*)(pF+16)); // vector load 8x8-bit						
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_3 = _mm_add_epi32(sum_3, lower_prods);
				sum_4 = _mm_add_epi32(sum_4, lower_prods_2);				
			}							
			pF +=24;
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
						load1_3 = _mm_cvtepu8_epi16(load1_3);		
						
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i diff16s_3 = _mm_sub_epi16(load1_3, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
				__m128i sqrdiff16s_3 = _mm_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)		
				        sum_3 = _mm_add_epi32(sum_3, upper_prods); 
				        sum_4 = _mm_add_epi32(sum_4, upper_prods_2); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				__m128i prod_low_3 = _mm_mullo_epi16(sqrdiff16s_3, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_3 = _mm_mulhi_epu16(sqrdiff16s_3, load3_1); // 8x16-bit mul (hi part)		
				
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				__m128i lower_prods_3 = _mm_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
				__m128i upper_prods_3 = _mm_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
				        load1        = _mm_loadu_si128((__m128i*)pF); 

				sum_5 				= _mm_add_epi32(sum_5, lower_prods); 
				sum_6 				= _mm_add_epi32(sum_6, lower_prods_2); 
				sum_7 				= _mm_add_epi32(sum_7, lower_prods_3); 
				sum_5 				= _mm_add_epi32(sum_5, upper_prods); 
				sum_6 				= _mm_add_epi32(sum_6, upper_prods_2); 
				sum_7 				= _mm_add_epi32(sum_7, upper_prods_3); 
			}							
		}
		sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0xee)); // horizontal 32-bit add
		sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0xee)); // horizontal 32-bit add			
		sum_1 = _mm_add_epi32(sum_1, _mm_shuffle_epi32(sum_1, 0x55)); // horizontal 32-bit add
		sum_2 = _mm_add_epi32(sum_2, _mm_shuffle_epi32(sum_2, 0x55)); // horizontal 32-bit add
		sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0xee)); // horizontal 32-bit add
		sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0xee)); // horizontal 32-bit add			
		sum_3 = _mm_add_epi32(sum_3, _mm_shuffle_epi32(sum_3, 0x55)); // horizontal 32-bit add
		sum_4 = _mm_add_epi32(sum_4, _mm_shuffle_epi32(sum_4, 0x55)); // horizontal 32-bit add
		sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0xee)); // horizontal 32-bit add
		sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0xee)); // horizontal 32-bit add		
		sum_5 = _mm_add_epi32(sum_5, _mm_shuffle_epi32(sum_5, 0x55)); // horizontal 32-bit add
		sum_6 = _mm_add_epi32(sum_6, _mm_shuffle_epi32(sum_6, 0x55)); // horizontal 32-bit add
		sum_7 = _mm_add_epi32(sum_7, _mm_shuffle_epi32(sum_7, 0xee)); // horizontal 32-bit add
		sum_7 = _mm_add_epi32(sum_7, _mm_shuffle_epi32(sum_7, 0x55)); // horizontal 32-bit add

		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_1) + (uint64_t)*pC;
		MinScores[0]    = Sum64u < MinScores[0] ? Sum64u : MinScores[0];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_2) + (uint64_t)*pC;
		MinScores[1]    = Sum64u < MinScores[1] ? Sum64u : MinScores[1];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_3) + (uint64_t)*pC;
		MinScores[2]    = Sum64u < MinScores[2] ? Sum64u : MinScores[2];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_4) + (uint64_t)*pC;
		MinScores[3]    = Sum64u < MinScores[3] ? Sum64u : MinScores[3];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_5) + (uint64_t)*pC;
		MinScores[4]    = Sum64u < MinScores[4] ? Sum64u : MinScores[4];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_6) + (uint64_t)*pC;
		MinScores[5]    = Sum64u < MinScores[5] ? Sum64u : MinScores[5];
		Sum64u          = (uint64_t)_mm_cvtsi128_si32(sum_7) + (uint64_t)*pC;
		MinScores[6]    = Sum64u < MinScores[6] ? Sum64u : MinScores[6];

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	pScores[0]      = MinScores[0];
	pScores[1]      = MinScores[1];
	pScores[2]      = MinScores[2];
	pScores[3]      = MinScores[3];
	pScores[4]      = MinScores[4];
	pScores[5]      = MinScores[5];
	pScores[6]      = MinScores[6];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g8_sse4
void
gmm_maxmix_8u8u_32u_g8(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t  GConst64u;
	uint64_t  Scores64u[2];	
	uint64_t  MinScore = ScoreLimit32u;
	__m128i MinScores_1 = CVT64_128((__m128i*) &MinScore); 	
	        MinScores_1 = _mm_shuffle_epi32(MinScores_1, _MM_SHUFFLE(1,0,1,0));
	__m128i MinScores_2 = MinScores_1;
	__m128i MinScores_3 = MinScores_1;
	__m128i MinScores_4 = MinScores_1;
	
	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		
		__m128i sum_1         = _mm_setzero_si128();
		__m128i sum_2         = _mm_setzero_si128();		
		__m128i sum_3         = _mm_setzero_si128();
		__m128i sum_4         = _mm_setzero_si128();		
		__m128i sum_5         = _mm_setzero_si128();
		__m128i sum_6         = _mm_setzero_si128();		
		__m128i sum_7         = _mm_setzero_si128();
		__m128i sum_8         = _mm_setzero_si128();		
			
		__m128i upper_prods;		
		__m128i upper_prods_2;		
		__m128i load1         = _mm_load_si128((__m128i*)pF); 
		__m128i load2         = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3         = CVT64_128((__m128i*)pV); // vector load 8x8-bit						

		for(j = 8; j <= nVecElements; j+=8)
		{			
			pF += 16;
			__m128i load2_1 = _mm_cvtepu8_epi16(load2); 			
			__m128i load3_1 = _mm_cvtepu8_epi16(load3);												
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
		                load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
		                load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit						
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        load1         = _mm_load_si128((__m128i*)pF); 
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods   = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_1 = _mm_add_epi32(sum_1, lower_prods);
				sum_2 = _mm_add_epi32(sum_2, lower_prods_2);				
			}
			pF += 16;									
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
				__m128i	load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
   					    sum_1 = _mm_add_epi32(sum_1, upper_prods); 
				        sum_2 = _mm_add_epi32(sum_2, upper_prods_2); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
						load1 = _mm_load_si128((__m128i*)pF); 								
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_3 = _mm_add_epi32(sum_3, lower_prods);
				sum_4 = _mm_add_epi32(sum_4, lower_prods_2);				
			}				
			pF += 16;	
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
						load1 = _mm_load_si128((__m128i*)pF); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
				        sum_3 = _mm_add_epi32(sum_3, upper_prods); 
				        sum_4 = _mm_add_epi32(sum_4, upper_prods_2); 
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				        upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				        upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd				        
				sum_5 = _mm_add_epi32(sum_5, lower_prods);
				sum_6 = _mm_add_epi32(sum_6, lower_prods_2);				
			}
			pF += 16;	
			{
				__m128i load1_2 = _mm_shuffle_epi32(load1, _MM_SHUFFLE(3,2,3,2));
                __m128i load1_1 = _mm_cvtepu8_epi16(load1); 
						load1_2 = _mm_cvtepu8_epi16(load1_2);		
			
				__m128i diff16s = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit		
				__m128i diff16s_2 = _mm_sub_epi16(load1_2, load2_1); // convert to 8x16-bit		
				__m128i sqrdiff16s = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
				__m128i sqrdiff16s_2 = _mm_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)		
				        sum_5 = _mm_add_epi32(sum_5, upper_prods); 
				        sum_6 = _mm_add_epi32(sum_6, upper_prods_2); 
				__m128i prod_low = _mm_mullo_epi16(sqrdiff16s, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high = _mm_mulhi_epu16(sqrdiff16s, load3_1); // 8x16-bit mul (hi part)
				__m128i prod_low_2 = _mm_mullo_epi16(sqrdiff16s_2, load3_1); // 8x16-bit mult (lo part)
				__m128i prod_high_2 = _mm_mulhi_epu16(sqrdiff16s_2, load3_1); // 8x16-bit mul (hi part)		
						load1 = _mm_load_si128((__m128i*)pF); 
				__m128i lower_prods = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				__m128i lower_prods_2 = _mm_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
				__m128i upper_prods_2 = _mm_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
				sum_7 = _mm_add_epi32(sum_7, lower_prods);
				sum_8 = _mm_add_epi32(sum_8, lower_prods_2);				
				sum_7 = _mm_add_epi32(sum_7, upper_prods); 
				sum_8 = _mm_add_epi32(sum_8, upper_prods_2); 
			}
		}
				GConst64u       = *pC; 
				__m128i gconst  = CVT64_128((__m128i*)&GConst64u);
				gconst          = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1,0,1,0));
				sum_1			= _mm_hadd_epi32(sum_1, sum_2);
				sum_2           = _mm_hadd_epi32(sum_3, sum_4);
				sum_3           = _mm_hadd_epi32(sum_5, sum_6);
				sum_4           = _mm_hadd_epi32(sum_7, sum_8);
				sum_1           = _mm_hadd_epi32(sum_1, sum_2);
				sum_2           = _mm_shuffle_epi32(sum_1, _MM_SHUFFLE(3,2,3,2));
				sum_3           = _mm_hadd_epi32(sum_3, sum_4);
				sum_4           = _mm_shuffle_epi32(sum_3, _MM_SHUFFLE(3,2,3,2));
				sum_1           = _mm_cvtepu32_epi64(sum_1);		
				sum_2           = _mm_cvtepu32_epi64(sum_2);		
				sum_3           = _mm_cvtepu32_epi64(sum_3);		
				sum_4           = _mm_cvtepu32_epi64(sum_4);		
				sum_1           = _mm_add_epi64(sum_1, gconst);
				sum_2           = _mm_add_epi64(sum_2, gconst);
				sum_3           = _mm_add_epi64(sum_3, gconst);
				sum_4           = _mm_add_epi64(sum_4, gconst);

			__m128i cmpres_1    = _mm_cmpgt_epi64(MinScores_1, sum_1); 
			__m128i cmpres_2    = _mm_cmpgt_epi64(MinScores_2, sum_2); 
			__m128i cmpres_3    = _mm_cmpgt_epi64(MinScores_3, sum_3); 
			__m128i cmpres_4    = _mm_cmpgt_epi64(MinScores_4, sum_4); 
			__m128i res_min_1   = _mm_andnot_si128(cmpres_1, MinScores_1); 
			__m128i res_min_2   = _mm_andnot_si128(cmpres_2, MinScores_2); 
			__m128i res_min_3   = _mm_andnot_si128(cmpres_3, MinScores_3); 
			__m128i res_min_4   = _mm_andnot_si128(cmpres_4, MinScores_4); 
			__m128i res_sum_1   = _mm_and_si128(cmpres_1, sum_1); 
			__m128i res_sum_2   = _mm_and_si128(cmpres_2, sum_2); 
			__m128i res_sum_3   = _mm_and_si128(cmpres_3, sum_3); 
			__m128i res_sum_4   = _mm_and_si128(cmpres_4, sum_4); 
					MinScores_1 = _mm_or_si128(res_min_1, res_sum_1);
					MinScores_2 = _mm_or_si128(res_min_2, res_sum_2);
					MinScores_3 = _mm_or_si128(res_min_3, res_sum_3);
					MinScores_4 = _mm_or_si128(res_min_4, res_sum_4);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}

	_mm_storeu_si128((__m128i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[1];
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_2);
	pScores[2]      = Scores64u[0];
	pScores[3]      = Scores64u[1];
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_3);
	pScores[4]      = Scores64u[0];
	pScores[5]      = Scores64u[1];
	_mm_storeu_si128((__m128i*)Scores64u, MinScores_4);
	pScores[6]      = Scores64u[0];
	pScores[7]      = Scores64u[1];
}

#endif //#if (OPT_LEVEL > 1 && OPT_LEVEL < 6) // SSE4/AVX1 only 

#if OPT_LEVEL > 5 // AVX2 +

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g2_avx2
void
gmm_maxmix_8u8u_32u_g2(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t  i,j;
	uint64_t  GConst64;
	uint64_t  Scores64u[2];
	uint64_t  MinScore = ScoreLimit32u;

	__m128i MinScores = CVT64_128((__m128i*) &MinScore); 	
	        MinScores = _mm_shuffle_epi32(MinScores, _MM_SHUFFLE(1,0,1,0));

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;

		__m256i sum_1             = _mm256_setzero_si256();

		__m128i load1             = _mm_loadu_si128((__m128i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		__m256i load256_1         = _mm256_cvtepu8_epi16(load1); 	
		for(j = 8; j <= nVecElements; j+=8)
		{	
				pF += 16;									
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												

			load256_2             = _mm256_permute4x64_epi64(load256_2, 0x44);
			load256_3             = _mm256_permute4x64_epi64(load256_3, 0x44);
	
			__m256i diff16s       = _mm256_sub_epi16(load256_1, load256_2); // convert to 8x16-bit		
			        load1         = _mm_loadu_si128((__m128i*)pF); 					
			__m256i sqrdiff16s    = _mm256_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)			
			 
		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		
			__m256i prod_low      = _mm256_mullo_epi16(sqrdiff16s, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high     = _mm256_mulhi_epu16(sqrdiff16s, load256_3); // 8x16-bit mul (hi part)
			        load256_1     = _mm256_cvtepu8_epi16(load1); 	
			__m256i lower_prods   = _mm256_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
			__m256i upper_prods   = _mm256_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
					
			sum_1 				  = _mm256_add_epi32(sum_1, lower_prods); 
			sum_1 				  = _mm256_add_epi32(sum_1, upper_prods); 
			}							
		GConst64 = *pC; 

		__m128i sum_f1  = _mm256_castsi256_si128(sum_1); 
		__m128i sum_f2  = _mm256_extractf128_si256(sum_1, 1); 
		__m128i gconst  = CVT64_128((__m128i*)&GConst64);
			    gconst  = _mm_shuffle_epi32(gconst, _MM_SHUFFLE(1,0,1,0));
		
		__m128i sum_64  = _mm_hadd_epi32(sum_f1, sum_f2);
				sum_64  = _mm_hadd_epi32(sum_64, sum_64);
				sum_64  = _mm_cvtepu32_epi64(sum_64);
				sum_64  = _mm_add_epi64(sum_64, gconst);
		__m128i cmpres  = _mm_cmpgt_epi64(MinScores, sum_64); 
		__m128i res_min = _mm_andnot_si128(cmpres, MinScores); 
		__m128i res_sum = _mm_and_si128(cmpres, sum_64); 
		MinScores = _mm_or_si128(res_min, res_sum);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
		}
	_mm_storeu_si128((__m128i*)Scores64u, MinScores);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[1];	
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g3_avx2
void
gmm_maxmix_8u8u_32u_g3(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t GConst64;
	uint64_t Scores64u[4];

	// init min scores
	uint64_t  MinScore = ScoreLimit32u;
	__m256i MinScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &MinScore)); 	

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		__m256i sum_1             = _mm256_setzero_si256();
		__m256i sum_2             = _mm256_setzero_si256();

		__m256i load1             = _mm256_loadu_si256((__m256i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						

		for(j = 8; j <= nVecElements; j+=8)
		{	
			pF += 24;
			__m256i load256_1_2   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_1   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_2   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2)); 	
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												

			        load256_2     = _mm256_permute4x64_epi64(load256_2, 0x44);
			        load256_3     = _mm256_permute4x64_epi64(load256_3, 0x44);
	
			__m256i diff16s_1     = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit		
			__m256i diff16s_2     = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_1  = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_2  = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)			

			__m256i prod_low_1    = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_1   = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_2    = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_2   = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

			        load1         = _mm256_loadu_si256((__m256i*)pF); 

			__m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
			__m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
			__m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
			
		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		

			        sum_1 		  = _mm256_add_epi32(sum_1, lower_prods_1); 
					sum_2 		  = _mm256_add_epi32(sum_2, lower_prods_2); 
					sum_1 		  = _mm256_add_epi32(sum_1, upper_prods_1); 
			        sum_2 		  = _mm256_add_epi32(sum_2, upper_prods_2); 
		}

		GConst64 = *pC; 

		        sum_1   = _mm256_hadd_epi32(sum_1, sum_2);
		__m128i sum_f1  = _mm256_castsi256_si128(sum_1); 
		         sum_1  = _mm256_permute4x64_epi64(sum_1, 0xee);
		__m256i gconst  = _mm256_broadcastq_epi64 (CVT64_128((__m128i*)&GConst64));
		__m128i sum_f2  = _mm256_castsi256_si128(sum_1);
		__m256i sum_64  = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
				sum_64  = _mm256_add_epi64(sum_64, gconst);
		__m256i cmpres  = _mm256_cmpgt_epi64(MinScores, sum_64); 
		__m256i res_min = _mm256_andnot_si256(cmpres, MinScores); 
		__m256i res_sum = _mm256_and_si256(cmpres, sum_64); 
		MinScores = _mm256_or_si256(res_min, res_sum);
		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	_mm256_storeu_si256((__m256i*)Scores64u, MinScores);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[2];
	pScores[2]      = Scores64u[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g4_avx2
void
gmm_maxmix_8u8u_32u_g4(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t GConst64;
	uint64_t Scores64u[4];

	// init min scores
	uint64_t  MinScore = ScoreLimit32u;
	__m256i MinScores = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &MinScore)); 	

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		__m256i sum_1             = _mm256_setzero_si256();
		__m256i sum_2             = _mm256_setzero_si256();

		__m256i load1             = _mm256_loadu_si256((__m256i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		
		for(j = 8; j <= nVecElements; j+=8)
		{	
			pF += 32;
			__m256i load256_1_2   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_1   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_2   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2)); 	
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												

			        load256_2     = _mm256_permute4x64_epi64(load256_2, 0x44);
			        load256_3     = _mm256_permute4x64_epi64(load256_3, 0x44);
	
			__m256i diff16s_1     = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit		
			__m256i diff16s_2     = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_1  = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_2  = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)			

			__m256i prod_low_1    = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_1   = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_2    = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_2   = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

			        load1         = _mm256_loadu_si256((__m256i*)pF); 

			__m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
			__m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
			__m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
			
		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		

			        sum_1 		  = _mm256_add_epi32(sum_1, lower_prods_1); 
					sum_2 		  = _mm256_add_epi32(sum_2, lower_prods_2); 
					sum_1 		  = _mm256_add_epi32(sum_1, upper_prods_1); 
			        sum_2 		  = _mm256_add_epi32(sum_2, upper_prods_2); 
		}

		GConst64 = *pC; 

		        sum_1   = _mm256_hadd_epi32(sum_1, sum_2);
		__m128i sum_f1  = _mm256_castsi256_si128(sum_1); 
		         sum_1  = _mm256_permute4x64_epi64(sum_1, 0xee);
		__m256i gconst  = _mm256_broadcastq_epi64 (CVT64_128((__m128i*)&GConst64));
		__m128i sum_f2  = _mm256_castsi256_si128(sum_1);
		__m256i sum_64  = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
				sum_64  = _mm256_add_epi64(sum_64, gconst);
		__m256i cmpres  = _mm256_cmpgt_epi64(MinScores, sum_64); 
		__m256i res_min = _mm256_andnot_si256(cmpres, MinScores); 
		__m256i res_sum = _mm256_and_si256(cmpres, sum_64); 
		MinScores = _mm256_or_si256(res_min, res_sum);
		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}

	_mm256_storeu_si256((__m256i*)Scores64u, MinScores);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[2];
	pScores[2]      = Scores64u[1];
	pScores[3]      = Scores64u[3];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g5_avx2
void
gmm_maxmix_8u8u_32u_g5(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t GConst64;
	uint64_t Scores64u[4];
	// init min scores
	uint64_t  MinScore = ScoreLimit32u;
	__m256i MinScores_1 = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &MinScore)); 	
	__m256i MinScores_2 = MinScores_1;

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		__m256i sum_1             = _mm256_setzero_si256();
		__m256i sum_2             = _mm256_setzero_si256();
		__m256i sum_3             = _mm256_setzero_si256();

		__m256i load1             = _mm256_loadu_si256((__m256i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		
		for(j = 8; j <= nVecElements; j+=8)
		{	
			pF += 32;
			__m256i load256_1_2   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_1   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_2   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2)); 	
			__m256i load256_1_3   = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)pF)); 
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												
			
			        load256_2     = _mm256_permute4x64_epi64(load256_2, 0x44);
			        load256_3     = _mm256_permute4x64_epi64(load256_3, 0x44);
	
			__m256i diff16s_1     = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit		
			__m256i diff16s_2     = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit		
			__m256i diff16s_3     = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_1  = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_2  = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_3  = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)			

			pF += 8;

			__m256i prod_low_1    = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_1   = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_2    = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_2   = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_3    = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_3   = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
			        load1         = _mm256_loadu_si256((__m256i*)pF); 

			__m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
			__m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
			__m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
			__m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
			__m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
			
		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		

			        sum_1 		  = _mm256_add_epi32(sum_1, lower_prods_1); 
					sum_2 		  = _mm256_add_epi32(sum_2, lower_prods_2); 
					sum_3 		  = _mm256_add_epi32(sum_3, lower_prods_3); 
					sum_1 		  = _mm256_add_epi32(sum_1, upper_prods_1); 
			        sum_2 		  = _mm256_add_epi32(sum_2, upper_prods_2); 
					sum_3 		  = _mm256_add_epi32(sum_3, upper_prods_3); 
		}

		GConst64 = *pC; 

		        sum_1     = _mm256_hadd_epi32(sum_1, sum_2);
				sum_3     = _mm256_hadd_epi32(sum_3, sum_3);
		__m128i sum_f1    = _mm256_castsi256_si128(sum_1); 
		__m128i sum_f3    = _mm256_castsi256_si128(sum_3); 
		         sum_1    = _mm256_permute4x64_epi64(sum_1, 0xee);
				 sum_3    = _mm256_permute4x64_epi64(sum_3, 0xee);
		__m256i gconst    = _mm256_broadcastq_epi64 (CVT64_128((__m128i*)&GConst64));
		__m128i sum_f2    = _mm256_castsi256_si128(sum_1);
		__m128i sum_f4    = _mm256_castsi256_si128(sum_3);
		__m256i sum_64_1  = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
		__m256i sum_64_2  = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
				sum_64_1  = _mm256_add_epi64(sum_64_1, gconst);
				sum_64_2  = _mm256_add_epi64(sum_64_2, gconst);
		__m256i cmpres_1  = _mm256_cmpgt_epi64(MinScores_1, sum_64_1); 
		__m256i cmpres_2  = _mm256_cmpgt_epi64(MinScores_2, sum_64_2); 
		__m256i res_min_1 = _mm256_andnot_si256(cmpres_1, MinScores_1); 
		__m256i res_min_2 = _mm256_andnot_si256(cmpres_2, MinScores_2); 
		__m256i res_sum_1 = _mm256_and_si256(cmpres_1, sum_64_1); 
		__m256i res_sum_2 = _mm256_and_si256(cmpres_2, sum_64_2); 
		MinScores_1       = _mm256_or_si256(res_min_1, res_sum_1);
		MinScores_2       = _mm256_or_si256(res_min_2, res_sum_2);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}

	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[2];
	pScores[2]      = Scores64u[1];
	pScores[3]      = Scores64u[3];
	pScores[4]      = _mm_cvtsi128_si32(_mm256_castsi256_si128(MinScores_2));
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g6_avx2
void
gmm_maxmix_8u8u_32u_g6(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t GConst64;
	uint64_t Scores64u[4];
	// init min scores
	uint64_t  MinScore = ScoreLimit32u;
	__m256i MinScores_1 = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &MinScore)); 	
	__m256i MinScores_2 = MinScores_1;

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;
		__m256i sum_1             = _mm256_setzero_si256();
		__m256i sum_2             = _mm256_setzero_si256();
		__m256i sum_3             = _mm256_setzero_si256();

		__m256i load1             = _mm256_loadu_si256((__m256i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		
		for(j = 8; j <= nVecElements; j+=8)
		{	
			pF += 32;
			__m256i load256_1_2   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_1   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_2   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2)); 	
			__m256i load256_1_3   = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)pF)); 
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												
			
			        load256_2     = _mm256_permute4x64_epi64(load256_2, 0x44);
			        load256_3     = _mm256_permute4x64_epi64(load256_3, 0x44);
	
			__m256i diff16s_1     = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit		
			__m256i diff16s_2     = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit		
			__m256i diff16s_3     = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_1  = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_2  = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_3  = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)			

			pF += 16;

			__m256i prod_low_1    = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_1   = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_2    = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_2   = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_3    = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_3   = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
			        load1         = _mm256_loadu_si256((__m256i*)pF); 

			__m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
			__m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
			__m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
			__m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
			__m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
			
		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		

			        sum_1 		  = _mm256_add_epi32(sum_1, lower_prods_1); 
					sum_2 		  = _mm256_add_epi32(sum_2, lower_prods_2); 
					sum_3 		  = _mm256_add_epi32(sum_3, lower_prods_3); 
					sum_1 		  = _mm256_add_epi32(sum_1, upper_prods_1); 
			        sum_2 		  = _mm256_add_epi32(sum_2, upper_prods_2); 
					sum_3 		  = _mm256_add_epi32(sum_3, upper_prods_3); 
		}

		GConst64 = *pC; 

		        sum_1     = _mm256_hadd_epi32(sum_1, sum_2);
				sum_3     = _mm256_hadd_epi32(sum_3, sum_3);
		__m128i sum_f1    = _mm256_castsi256_si128(sum_1); 
		__m128i sum_f3    = _mm256_castsi256_si128(sum_3); 
		         sum_1    = _mm256_permute4x64_epi64(sum_1, 0xee);
				 sum_3    = _mm256_permute4x64_epi64(sum_3, 0xee);
		__m256i gconst    = _mm256_broadcastq_epi64 (CVT64_128((__m128i*)&GConst64));
		__m128i sum_f2    = _mm256_castsi256_si128(sum_1);
		__m128i sum_f4    = _mm256_castsi256_si128(sum_3);
		__m256i sum_64_1  = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
		__m256i sum_64_2  = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
				sum_64_1  = _mm256_add_epi64(sum_64_1, gconst);
				sum_64_2  = _mm256_add_epi64(sum_64_2, gconst);
		__m256i cmpres_1  = _mm256_cmpgt_epi64(MinScores_1, sum_64_1); 
		__m256i cmpres_2  = _mm256_cmpgt_epi64(MinScores_2, sum_64_2); 
		__m256i res_min_1 = _mm256_andnot_si256(cmpres_1, MinScores_1); 
		__m256i res_min_2 = _mm256_andnot_si256(cmpres_2, MinScores_2); 
		__m256i res_sum_1 = _mm256_and_si256(cmpres_1, sum_64_1); 
		__m256i res_sum_2 = _mm256_and_si256(cmpres_2, sum_64_2); 
		MinScores_1       = _mm256_or_si256(res_min_1, res_sum_1);
		MinScores_2       = _mm256_or_si256(res_min_2, res_sum_2);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}

	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[2];
	pScores[2]      = Scores64u[1];
	pScores[3]      = Scores64u[3];
	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_2);
	pScores[4]      = Scores64u[0];
	pScores[5]      = Scores64u[2];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g7_avx2
void
gmm_maxmix_8u8u_32u_g7(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t GConst64;
	uint64_t Scores64u[4];
	uint64_t  MinScore = ScoreLimit32u;
	__m256i MinScores_1 = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &MinScore)); 	
	__m256i MinScores_2 = MinScores_1;

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;

		__m256i sum_1             = _mm256_setzero_si256();
		__m256i sum_2             = _mm256_setzero_si256();
		__m256i sum_3             = _mm256_setzero_si256();
		__m256i sum_4             = _mm256_setzero_si256();

		__m256i load1             = _mm256_loadu_si256((__m256i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		
		for(j = 8; j <= nVecElements; j+=8)
		{	
			pF += 32;
			
			__m256i load256_1_2   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_1   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_2   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2)); 	
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												

			        load256_2     = _mm256_permute4x64_epi64(load256_2, 0x44);
			        load256_3     = _mm256_permute4x64_epi64(load256_3, 0x44);
			
			__m256i diff16s_1     = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit		
			__m256i diff16s_2     = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_1  = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_2  = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)			

					load1         = _mm256_loadu_si256((__m256i*)pF); 

			__m256i prod_low_1    = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_1   = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_2    = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_2   = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		
			        

			__m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
			__m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
			__m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
						
			        sum_1 		  = _mm256_add_epi32(sum_1, lower_prods_1); 
					sum_2 		  = _mm256_add_epi32(sum_2, lower_prods_2); 
					sum_1 		  = _mm256_add_epi32(sum_1, upper_prods_1); 
			        sum_2 		  = _mm256_add_epi32(sum_2, upper_prods_2); 
			
			pF += 24;
			
			__m256i load256_1_4   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_3   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_4   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_4)); 	
	
			
			__m256i diff16s_3     = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit		
			__m256i diff16s_4     = _mm256_sub_epi16(load256_1_4, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_3  = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_4  = _mm256_mullo_epi16(diff16s_4, diff16s_4); // 8x16-bit mul (hi zero)			

					load1         = _mm256_loadu_si256((__m256i*)pF); 
			__m256i prod_low_3    = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_3   = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_4    = _mm256_mullo_epi16(sqrdiff16s_4, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_4   = _mm256_mulhi_epu16(sqrdiff16s_4, load256_3); // 8x16-bit mul (hi part)

			        

			__m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
			__m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
			__m256i lower_prods_4 = _mm256_unpacklo_epi16(prod_low_4, prod_high_4); // lo 4x32-bit prd
			__m256i upper_prods_4 = _mm256_unpackhi_epi16(prod_low_4, prod_high_4); // hi 4x32-bit prd

			        sum_3 		  = _mm256_add_epi32(sum_3, lower_prods_3); 
					sum_4 		  = _mm256_add_epi32(sum_4, lower_prods_4); 
					sum_3 		  = _mm256_add_epi32(sum_3, upper_prods_3); 
			        sum_4 		  = _mm256_add_epi32(sum_4, upper_prods_4); 
			

		}
		GConst64 = *pC; 

		        sum_1    = _mm256_hadd_epi32(sum_1, sum_2);
				sum_3    = _mm256_hadd_epi32(sum_3, sum_4);
		__m128i sum_f1   = _mm256_castsi256_si128(sum_1); 
		__m128i sum_f3   = _mm256_castsi256_si128(sum_3); 
		         sum_1   = _mm256_permute4x64_epi64(sum_1, 0xee);
				 sum_3   = _mm256_permute4x64_epi64(sum_3, 0xee);
		__m256i gconst   = _mm256_broadcastq_epi64 (CVT64_128((__m128i*)&GConst64));
		__m128i sum_f2   = _mm256_castsi256_si128(sum_1);
		__m128i sum_f4   = _mm256_castsi256_si128(sum_3);

		__m256i sum_64_1 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
				sum_64_1 = _mm256_add_epi64(sum_64_1, gconst);
		__m256i sum_64_2 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
				sum_64_2 = _mm256_add_epi64(sum_64_2, gconst);

		__m256i cmpres_1 = _mm256_cmpgt_epi64(MinScores_1, sum_64_1); 
		__m256i cmpres_2 = _mm256_cmpgt_epi64(MinScores_2, sum_64_2); 
		__m256i res_min_1  = _mm256_andnot_si256(cmpres_1, MinScores_1); 
		__m256i res_min_2  = _mm256_andnot_si256(cmpres_2, MinScores_2); 
		__m256i res_sum_1  = _mm256_and_si256(cmpres_1, sum_64_1); 
		__m256i res_sum_2  = _mm256_and_si256(cmpres_2, sum_64_2); 
		MinScores_1 = _mm256_or_si256(res_min_1, res_sum_1);
		MinScores_2 = _mm256_or_si256(res_min_2, res_sum_2);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}

	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[2];
	pScores[2]      = Scores64u[1];
	pScores[3]      = Scores64u[3];
	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_2);
	pScores[4]      = Scores64u[0];
	pScores[5]      = Scores64u[2];
	pScores[6]      = Scores64u[1];
}

//gmm_maxmix_8u8u_32u_grouped_opt_f8_g8_avx2
void
gmm_maxmix_8u8u_32u_g8(
	_GMM8_MAXMIX_ARGS)
{
	const uint8_t *pM = pMeans;
	const uint8_t *pV = pVars;
	const uint8_t *pF = pFeat;
	const uint32_t *pC = pGconst;
	uint32_t i,j;
	uint64_t GConst64;
	uint64_t Scores64u[4];
	uint64_t  MinScore = ScoreLimit32u;
	__m256i MinScores_1 = _mm256_broadcastq_epi64(CVT64_128((__m128i*) &MinScore)); 	
	__m256i MinScores_2 = MinScores_1;

	for(i = 0; i < nMixtures; i++)
	{
		pF = pFeat;

		__m256i sum_1             = _mm256_setzero_si256();
		__m256i sum_2             = _mm256_setzero_si256();
		__m256i sum_3             = _mm256_setzero_si256();
		__m256i sum_4             = _mm256_setzero_si256();

		__m256i load1             = _mm256_loadu_si256((__m256i*)pF); 
		__m128i load2             = CVT64_128((__m128i*)pM); // vector load 8x8-bit
		__m128i load3             = CVT64_128((__m128i*)pV); // vector load 8x8-bit						
		
		for(j = 8; j <= nVecElements; j+=8)
		{	
			pF += 32;
			
			__m256i load256_1_2   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_1   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_2   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_2)); 	
			__m256i load256_2     = _mm256_cvtepu8_epi16(load2); 			
			__m256i load256_3     = _mm256_cvtepu8_epi16(load3);												

			        load256_2     = _mm256_permute4x64_epi64(load256_2, 0x44);
			        load256_3     = _mm256_permute4x64_epi64(load256_3, 0x44);
			
			__m256i diff16s_1     = _mm256_sub_epi16(load256_1_1, load256_2); // convert to 8x16-bit		
			__m256i diff16s_2     = _mm256_sub_epi16(load256_1_2, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_1  = _mm256_mullo_epi16(diff16s_1, diff16s_1); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_2  = _mm256_mullo_epi16(diff16s_2, diff16s_2); // 8x16-bit mul (hi zero)			

					load1         = _mm256_loadu_si256((__m256i*)pF); 

			__m256i prod_low_1    = _mm256_mullo_epi16(sqrdiff16s_1, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_1   = _mm256_mulhi_epu16(sqrdiff16s_1, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_2    = _mm256_mullo_epi16(sqrdiff16s_2, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_2   = _mm256_mulhi_epu16(sqrdiff16s_2, load256_3); // 8x16-bit mul (hi part)

		            load2         = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit
					load3         = CVT64_128((__m128i*)&pV[j]); // vector load 8x8-bit		
			        

			__m256i lower_prods_1 = _mm256_unpacklo_epi16(prod_low_1, prod_high_1); // lo 4x32-bit prd
			__m256i upper_prods_1 = _mm256_unpackhi_epi16(prod_low_1, prod_high_1); // hi 4x32-bit prd
			__m256i lower_prods_2 = _mm256_unpacklo_epi16(prod_low_2, prod_high_2); // lo 4x32-bit prd
			__m256i upper_prods_2 = _mm256_unpackhi_epi16(prod_low_2, prod_high_2); // hi 4x32-bit prd
						
			        sum_1 		  = _mm256_add_epi32(sum_1, lower_prods_1); 
					sum_2 		  = _mm256_add_epi32(sum_2, lower_prods_2); 
					sum_1 		  = _mm256_add_epi32(sum_1, upper_prods_1); 
			        sum_2 		  = _mm256_add_epi32(sum_2, upper_prods_2); 
			
			pF += 32;
			
			__m256i load256_1_4   = _mm256_permute4x64_epi64(load1, 0xee);
			__m256i load256_1_3   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load1)); 	
			        load256_1_4   = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(load256_1_4)); 	
	
			
			__m256i diff16s_3     = _mm256_sub_epi16(load256_1_3, load256_2); // convert to 8x16-bit		
			__m256i diff16s_4     = _mm256_sub_epi16(load256_1_4, load256_2); // convert to 8x16-bit		

			__m256i sqrdiff16s_3  = _mm256_mullo_epi16(diff16s_3, diff16s_3); // 8x16-bit mul (hi zero)			
			__m256i sqrdiff16s_4  = _mm256_mullo_epi16(diff16s_4, diff16s_4); // 8x16-bit mul (hi zero)			

					load1         = _mm256_loadu_si256((__m256i*)pF); 
			__m256i prod_low_3    = _mm256_mullo_epi16(sqrdiff16s_3, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_3   = _mm256_mulhi_epu16(sqrdiff16s_3, load256_3); // 8x16-bit mul (hi part)
			__m256i prod_low_4    = _mm256_mullo_epi16(sqrdiff16s_4, load256_3); // 8x16-bit mult (lo part)
			__m256i prod_high_4   = _mm256_mulhi_epu16(sqrdiff16s_4, load256_3); // 8x16-bit mul (hi part)

			        

			__m256i lower_prods_3 = _mm256_unpacklo_epi16(prod_low_3, prod_high_3); // lo 4x32-bit prd
			__m256i upper_prods_3 = _mm256_unpackhi_epi16(prod_low_3, prod_high_3); // hi 4x32-bit prd
			__m256i lower_prods_4 = _mm256_unpacklo_epi16(prod_low_4, prod_high_4); // lo 4x32-bit prd
			__m256i upper_prods_4 = _mm256_unpackhi_epi16(prod_low_4, prod_high_4); // hi 4x32-bit prd

			        sum_3 		  = _mm256_add_epi32(sum_3, lower_prods_3); 
					sum_4 		  = _mm256_add_epi32(sum_4, lower_prods_4); 
					sum_3 		  = _mm256_add_epi32(sum_3, upper_prods_3); 
			        sum_4 		  = _mm256_add_epi32(sum_4, upper_prods_4); 
			

		}
		GConst64 = *pC; 

		        sum_1    = _mm256_hadd_epi32(sum_1, sum_2);
				sum_3    = _mm256_hadd_epi32(sum_3, sum_4);
		__m128i sum_f1   = _mm256_castsi256_si128(sum_1); 
		__m128i sum_f3   = _mm256_castsi256_si128(sum_3); 
		         sum_1   = _mm256_permute4x64_epi64(sum_1, 0xee);
				 sum_3   = _mm256_permute4x64_epi64(sum_3, 0xee);
		__m256i gconst   = _mm256_broadcastq_epi64 (CVT64_128((__m128i*)&GConst64));
		__m128i sum_f2   = _mm256_castsi256_si128(sum_1);
		__m128i sum_f4   = _mm256_castsi256_si128(sum_3);

		__m256i sum_64_1 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f1, sum_f2));
				sum_64_1 = _mm256_add_epi64(sum_64_1, gconst);
		__m256i sum_64_2 = _mm256_cvtepu32_epi64(_mm_hadd_epi32(sum_f3, sum_f4));
				sum_64_2 = _mm256_add_epi64(sum_64_2, gconst);

		__m256i cmpres_1 = _mm256_cmpgt_epi64(MinScores_1, sum_64_1); 
		__m256i cmpres_2 = _mm256_cmpgt_epi64(MinScores_2, sum_64_2); 
		__m256i res_min_1  = _mm256_andnot_si256(cmpres_1, MinScores_1); 
		__m256i res_min_2  = _mm256_andnot_si256(cmpres_2, MinScores_2); 
		__m256i res_sum_1  = _mm256_and_si256(cmpres_1, sum_64_1); 
		__m256i res_sum_2  = _mm256_and_si256(cmpres_2, sum_64_2); 
		MinScores_1 = _mm256_or_si256(res_min_1, res_sum_1);
		MinScores_2 = _mm256_or_si256(res_min_2, res_sum_2);

		pM += nVecElements;
		pV += nVecElements;
		pC++;
	}
	
	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_1);
	pScores[0]      = Scores64u[0];
	pScores[1]      = Scores64u[2];
	pScores[2]      = Scores64u[1];
	pScores[3]      = Scores64u[3];
	_mm256_storeu_si256((__m256i*)Scores64u, MinScores_2);
	pScores[4]      = Scores64u[0];
	pScores[5]      = Scores64u[2];
	pScores[6]      = Scores64u[1];
	pScores[7]      = Scores64u[3];
}

uint32_t
gmm_maxmix_8u16u_32u(
	_GMM16_ARGS)
{
  const uint8_t *pM = pMeans;
  const uint16_t *pV = pVars;
  const uint32_t *pC = pGconst;
    uint64_t Scores64u[2];
  uint64_t Sum64u;
  uint32_t i,j;
	uint64_t MinScore64u = ScoreLimit32u;
  
  for(i = 0; i < nMixtures; i++)
  {
		__m256i sum_1                = _mm256_setzero_si256();
		__m256i sum_2                = _mm256_setzero_si256();
		__m128i load1                = CVT64_128((__m128i*)pFeat);                                
		__m128i load2                = CVT64_128((__m128i*)pM); // vector load 8x8-bit                  
		__m128i load3                = _mm_loadu_si128((__m128i*)pV); // vector load 8x8-bit                                          
		for(j = 8; j <= nVecElements; j+=8)
    {

				__m128i load1_1      = _mm_cvtepu8_epi16(load1); 
				__m128i load2_1      = _mm_cvtepu8_epi16(load2);                 


				__m128i diff16s      = _mm_sub_epi16(load1_1, load2_1); // convert to 8x16-bit            
				__m128i sqrdiff16s   = _mm_mullo_epi16(diff16s, diff16s); // 8x16-bit mul (hi zero)             
				        load1        = CVT64_128((__m128i*)&pFeat[j]);                                
				        load2        = CVT64_128((__m128i*)&pM[j]); // vector load 8x8-bit                  

				__m128i prod_low     = _mm_mullo_epi16(sqrdiff16s, load3); // 8x16-bit mult (lo part)
				__m128i prod_high    = _mm_mulhi_epu16(sqrdiff16s, load3); // 8x16-bit mul (hi part)

				        load3        = _mm_loadu_si128((__m128i*)&pV[j]); // vector load 8x8-bit   

				__m128i lower_prods  = _mm_unpacklo_epi16(prod_low, prod_high); // lo 4x32-bit prd
				__m128i upper_prods  = _mm_unpackhi_epi16(prod_low, prod_high); // hi 4x32-bit prd
				
				        sum_1        = _mm256_add_epi64(sum_1, _mm256_cvtepu32_epi64(lower_prods));        
						sum_2        = _mm256_add_epi64(sum_2, _mm256_cvtepu32_epi64(upper_prods));        
    }

    
		sum_1           = _mm256_add_epi64(sum_1, sum_2);
		__m128i sum_f1  = _mm256_castsi256_si128(sum_1); 
		__m128i sum_f2  = _mm256_extractf128_si256(sum_1, 1); 
				sum_f1  = _mm_add_epi64(sum_f1, sum_f2);
		_mm_storeu_si128((__m128i*)Scores64u, sum_f1);
		Sum64u          = Scores64u[0] + Scores64u[1] + (uint64_t)*pC;

		MinScore64u     = Sum64u < MinScore64u ? Sum64u : MinScore64u;
  
    pM += nVecElements;
    pV += nVecElements;
    pC++;
  }
  
  return(MinScore64u);
}

#endif //#if OPT_LEVEL > 5 // AVX2+

GmmKernel KERNEL(gmmKernel) = 
{
    gmm_maxmix_8u8u_32u,
    gmm_maxmix_8u16u_32u,
#if OPT_LEVEL == 0 || OPT_LEVEL == 1
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
#else
    gmm_maxmix_8u8u_32u_g1,
    gmm_maxmix_8u8u_32u_g2,
    gmm_maxmix_8u8u_32u_g3,
    gmm_maxmix_8u8u_32u_g4,
    gmm_maxmix_8u8u_32u_g5,
    gmm_maxmix_8u8u_32u_g6,
    gmm_maxmix_8u8u_32u_g7,
    gmm_maxmix_8u8u_32u_g8
#endif
};
