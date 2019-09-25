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

#include "Cnn2DuArch.h"

#include "Bias.h"
#include "Component.h"
#include "ConvolutionalFunctions.h"
#include "ConvolutionalFunctions2D.h"
#include "PoolingFunctions2D.h"
#include "Tensor.h"

#include "gna-api-types-xnn.h"

#include <cmath>
#include <cstdint>
#include <memory>

#define uintDivCeil(dividend,divisor) (uint32_t)ceil((double)(dividend)/(double)(divisor))


namespace GNA
{

    // ------------------------------------------------------------------------------------------------
    // Constants & Typedef(s):
    // ------------------------------------------------------------------------------------------------
    // Arch Global Variables:


    static uint32_t  GNA3_NUM_CE = 8;      // Number of CE(s)
    static uint32_t  GNA3_UMEM_SIZE_KB = 32;     // Total Unified-Memory Size in KB
    static uint32_t  GNA3_NMEM_SIZE_KB = 2;      // Total Narrow-Mem Size in KB
    static float GNA3_BMEM_SIZE_KB = 0.5;    // Total Bias-Memory Size in KB
                                             // uArch Global Variables:
    static uint32_t GNA3_UBANK_ROW_SIZE_B = 16;     // Unified Memory Row Size (Bytes per EBB/RF Entry)
    static uint32_t GNA3_CONST_KB = 1024;   // 1KB = 1024 Bytes
                                         // uArch Feature(s):
    static bool GNA3_2DCNN_ALIST_EN = false;  // Enables Active-List as memory for 2D-CNN
    static bool GNA3_2DCNN_CNFLCT_EN = false;  // Allows conflicts in UMEM between KMEM and CMEM
    static bool GNA3_2DCNN_PLUPACK_EN = false;  // Enables Packing of PLU Elements, basically using size of ACTx, rather than 4B
    static bool GNA3_2DCNN_C1MEM_EN = false;  // CMEM_Elements = CNVw * 1 ; instead of CMEM_Elements = CNVw * Kh ; Only if (Kh * IFVw * IFVc) feets in Narrow-MEM

// uArch Hacks/Limitation Knobs(s):
    static bool GNA3_2DCNN_NMADJ = true;   // Narrow-Memory Adjusemt: When TRUE - Narrow-Mem is 16B Smaller (GNa-3.0 uArch RTL Rev 1.0)
    static bool GNA3_2DCNN_DMACNPREC = false;  // Dynamic-Mac-Precision: When TRUE - We can simulate with 4B per MAC
                                              // GNA UMEM Static Constatn(s):
    static uint32_t UMemSize = GNA3_UMEM_SIZE_KB * GNA3_CONST_KB;   // Total size of UMEM (Wide-Mem) in Bytes
    static uint32_t UMemRowSize = GNA3_NUM_CE * GNA3_UBANK_ROW_SIZE_B; // Total number of Bytes in a single UMEM Row (Accross all CEs)
    static uint32_t AListMemSize = (uint32_t)(2 * 0.25 * GNA3_CONST_KB);    // Size of Active-List Memory in Bytes
    static uint32_t XSize = (uint32_t)(0.25 * UMemSize);             // Size of Section-X of UMEM in Bytes
    static uint32_t YSize = (uint32_t)(0.25 * UMemSize);             // Size of Section-Y of UMEM in Bytes
    static uint32_t ZSize = (uint32_t)(0.50 * UMemSize);             // Size of Section-Z of UMEM in Bytes
    static uint32_t XYSize = XSize + YSize;                       // Size of Sections XY of UMEM in Bytes
    static uint32_t ZYSize = ZSize + YSize;                       // Size of Sections ZY of UMEM in Bytes


    typedef struct GNA3_Tensor // NHWC
    { 
        uint16_t    N;    // Number of 3D-Tensors (Used as 4th diemension)
        uint16_t    H;    // Height (Horizontal) Dimension
        uint16_t    W;    // Width (Vertical) Dimension
        uint16_t    C;    // Channel (Depth) Dimension
        gna_data_mode Prec; // Precision
    } GNA3_Tensor_t;

    static const GNA3_UMemAlloc GNA3_UMemAlloc_def;
    static const convolutional_fused_configuration gna_convolutional_fused_configuration_def;

    inline GNA3_Tensor_t GNA3_GetCNV(ConvolutionFunction2D const * cnnIn, const DataMode& outputMode)
    {
        GNA3_Tensor_t CNV = { };

        CNV.N = 1;
        CNV.H = static_cast<uint16_t>( 1 + (cnnIn->Input->at(GNA_DIM_H) + (2 * cnnIn->Padding->at(GNA_DIM_H)) - cnnIn->Filters->at(GNA_DIM_H)) / cnnIn->Stride->at(GNA_DIM_H) ); // Note: Using '/' operator for flooring
        CNV.W = static_cast<uint16_t>( 1 + (cnnIn->Input->at(GNA_DIM_W) + (2 * cnnIn->Padding->at(GNA_DIM_W)) - cnnIn->Filters->at(GNA_DIM_W)) / cnnIn->Stride->at(GNA_DIM_W) ); // Note: Using '/' operator for flooring
        CNV.C = static_cast<uint16_t>( cnnIn->Filters->Count);
        CNV.Prec = outputMode;
        return CNV;
    }

    inline GNA3_Tensor_t GNA3_GetPLV(ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn, const DataMode& outputMode)
    {
        GNA3_Tensor_t CNV = GNA3_GetCNV(cnnIn, outputMode);
        GNA3_Tensor_t PLV = { };

        if (poolingIn != nullptr && poolingIn->Mode != KernelPoolingModeNone)
        { // Pooling is enabled:
            PLV.N = 1;
            PLV.H = static_cast<uint16_t>(ceil(1 + double(CNV.H - poolingIn->Window->at(GNA_DIM_H)) / double(poolingIn->Stride->at(GNA_DIM_H))) );
            PLV.W = static_cast<uint16_t>(ceil(1 + double(CNV.W - poolingIn->Window->at(GNA_DIM_W)) / double(poolingIn->Stride->at(GNA_DIM_W))) );
            PLV.C = CNV.C;
            PLV.Prec = outputMode; //TODO: Check if valid
        }
        else
        { // Pooling is disabled (returns Null Tensor):
            PLV.N = PLV.H = PLV.W = PLV.C = 0;
            PLV.Prec = GNA_DATA_DISABLED;
        }
        return PLV;
    }

    inline uint32_t GNA3_KMemAllocSize(ConvolutionFunction2D const * cnnIn, convolutional_fused_configuration* const convConfiguration, uint32_t inPrec)
    {
        // GNA3_KMemAllocSize:
        // Returns the number of bytes needed to allocate a Kernel Into Unidifed-Memory
        // Depends on the uThreads, where more uThreds less residue/unused memory allocation
        // '-1' value returned, indicates an ERROR
        uint32_t KMemRowSizeB = GNA3_NUM_CE * GNA3_UBANK_ROW_SIZE_B / convConfiguration->uT;

        // Extracting CNN Params:
        uint32_t KPrec = cnnIn->Filters->Mode.Size;
        uint32_t KDimH = cnnIn->Filters->at(GNA_DIM_H);
        uint32_t KDimW = cnnIn->Filters->at(GNA_DIM_W);

        if ((KPrec == 1) && (inPrec == 2)) 
        {
            KPrec = 2;
        }

        uint32_t KSizeB = ALIGN(KPrec * KDimH * KDimW * cnnIn->Input->at(GNA_DIM_D) , KMemRowSizeB); // Net Kernel Size in Byte(s)
        return convConfiguration->KWG * KSizeB;
    }
    inline uint32_t uintFloorPwr2(const uint32_t Num) {
        // Static Mapping for Log2    Inp = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 }
        static const uint8_t log2Consts[] = { 0, 1, 2, 2, 4, 4, 4, 4, 8, 8,  8,  8,  8,  8,  8,  8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32 };
        return Num > 32 ? UINT32_MAX : log2Consts[Num];
    }
    inline uint32_t GNA3_UMemAllocSize(const uint32_t Elmnts, const uint32_t KWG, const uint32_t Prec)
    {

        // Function: GNA3_UMemAllocSize ; Returns Allocation size in UMEM in BYTEs.

        // Valid for CMEM & PMEM. (Following the UMEM uArch Allocation scheme)
        uint32_t UMemElmtsPerStack = GNA3_UBANK_ROW_SIZE_B / Prec;       // Number of Accumolators in a CMEM-Stack-Row (Single CE)
        uint32_t UMemSlots = GNA3_NUM_CE * UMemElmtsPerStack;

        uint32_t UMemAllocRows = 0;
        if (KWG <= UMemSlots / 2)
        { // AKA Stride-Packing (More than one stride in Physical UMEM-Row)
                                    // In Stride-Packing scheme, there are mulitple CNV strides in a single UMEM Row
            uint32_t SlotsPerKernel = uintFloorPwr2(UMemSlots / KWG);             // Notice '/' acts as flooring, which is OK in this case
            UMemAllocRows = uintDivCeil(Elmnts, SlotsPerKernel);      // Rows needed for entire KWG Kernels
        }
        else
        { // AKA Non-Stride-Packing
            uint32_t UMemRowsPerStride = uintDivCeil(KWG, UMemSlots);
            UMemAllocRows = UMemRowsPerStride * Elmnts;
        }
        return UMemRowSize * UMemAllocRows;
    }

    inline uint32_t GNA3_GetCnMemElmts(ConvolutionFunction2D const * cnnIn, const DataMode& outputMode)
    {
        GNA3_Tensor_t CNV = GNA3_GetCNV(cnnIn, outputMode);


        if ((GNA3_2DCNN_C1MEM_EN) && ((uint32_t)(cnnIn->Input->at(GNA_DIM_W) * cnnIn->Input->at(GNA_DIM_D) * cnnIn->Filters->at(GNA_DIM_H)) <= (uint32_t)(GNA3_NMEM_SIZE_KB * GNA3_CONST_KB)))
        {
            return CNV.W;
        }
        return CNV.W * cnnIn->Filters->at(GNA_DIM_H);
    }

    inline uint32_t GNA3_CMemAllocSize(ConvolutionFunction2D const * cnnIn, const DataMode& outputMode, convolutional_fused_configuration* const convConfiguration)
    {
        // Function: GNA3_CMemAllocSize ; <TODO> Description
        uint32_t            CnMemElmts = GNA3_GetCnMemElmts(cnnIn, outputMode);

        uint32_t MACPrec = (GNA3_2DCNN_DMACNPREC == false) ? 8 : 4;
        return GNA3_UMemAllocSize(CnMemElmts, convConfiguration->KWG, MACPrec); //HW only supports 8 MAC precission
    }

    inline uint32_t GNA3_GetPnMemElmts(ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn, const DataMode& outputMode)
    {
        GNA3_Tensor_t PLV = GNA3_GetPLV(cnnIn, poolingIn, outputMode);

        return PLV.W * (poolingIn != nullptr ? poolingIn->Window->at(GNA_DIM_H) : 0);
    }

    inline uint32_t GNA3_GetPLUEffPrec(PoolingFunction2D const * poolingIn, const DataMode& outputMode)
    {
        // When Pooling is enabled, PMEM is allocated onto the UMEM
        // Depends on the ACTx, and Pooling Type (MAX/SUM), an effecite Pooling Precision is used.
        // - For MAX Pooling: The Pooling Precision used in PMEM is SAME as the ACTx
        // - For SUM Pooling: PLU 'upgrades' the Precision to bigger/widther precision to support the SUM operation
        //
        // Follwoing Table depics the Effective-Precision used:
        //  ACTx   |   Pooling Type  |
        //  Prec   |   MAX  |   SUM  |
        // --------+--------+--------+
        //  DIS    | INT-32 | INT-32 |
        //  INT-8  | INT-8  | INT-16 |
        //  INT-16 | INT-16 | INT-32 |
        //  INT-32 | INT-32 | INT-32 |

        // If Pooling in not enabled, break here:
        if (poolingIn == nullptr) return 0;

        gna_data_mode PLUEffPrec = outputMode;

        if (poolingIn->Mode == KernelPoolingModeSum)
        {
            switch (outputMode)
            {
            case GNA_INT8:
                PLUEffPrec = GNA_INT16;
                break;
            case GNA_INT16:
                PLUEffPrec = GNA_INT32;
                break;
            default:
                PLUEffPrec = outputMode;
                break;
            }
        }

        return static_cast<uint32_t>(PLUEffPrec);
    }

    inline uint32_t GNA3_PMemAllocSize(ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn, const DataMode& outputMode, convolutional_fused_configuration* const convConfiguration)
    {
        // Function: GNA3_PMemAllocSize
        // Returns the PMemEmts, meaning the number of Elements in a single Pn allocation.
        uint32_t        PnMemElmts = GNA3_GetPnMemElmts(cnnIn, poolingIn, outputMode);
        uint32_t ACTx, PLUPrec;

        ACTx = outputMode.Size;

        if (!GNA3_2DCNN_PLUPACK_EN)
        { // TODO check if valid
            PLUPrec = 4;
        }
        else
        {
            PLUPrec = ACTx;
            PLUPrec = GNA3_GetPLUEffPrec(poolingIn, outputMode);
        }
        return GNA3_UMemAllocSize(PnMemElmts, convConfiguration->KWG, PLUPrec);
    }

    inline double dblUintEff(const uint32_t Numerator, const uint32_t Denominator)
    {
        double Div = ((double)Numerator / (double)Denominator);
        return Div / ceil(Div);
    }

    bool GNA3_GenAdaptHW_uT(ConvolutionFunction2D const * cnnIn, convolutional_fused_configuration* const convConfiguration)
    {
        // Calulating the Micro-Threads (uT) used by GNA-3.0 Adaptive-HW, for best Effciancy of GNA-HW
        // GNA's CVU (Convolutiopn-Unit) operates on a 'Kernel-Row' (Which is an extention to Channel-First)
        // Kernel-Row = IFVc * Kernel_DimW
        // MPT = MAC-Per-Thread (Number of Elements a Thread computes in single Cycle)

        // Extracting CNN Params:
        uint16_t KWG = convConfiguration->KWG;
        uint32_t KDimW = cnnIn->Filters->at(GNA_DIM_W);

        uint32_t DPWidth = ((cnnIn->Input->Mode == GNA_INT8 && cnnIn->Filters->Mode == GNA_INT8) ? 2 : 1);
        uint32_t TotalMAC = GNA3_NUM_CE * 8 * DPWidth;
        uint32_t KernelRow = cnnIn->Filters->at(GNA_DIM_D) * KDimW;

        double BestEff = 0;

        for (uint32_t uThreads = 1; ((uThreads <= GNA3_NUM_CE) && (uThreads <= KWG)); uThreads *= 2)
        {
            uint32_t   MPT = TotalMAC / uThreads;
            double uThreadEff = dblUintEff(KernelRow, MPT);
            double KWGEff = dblUintEff(KWG, uThreads);
            double StreamEff = ceil((double)KWG / (double)uThreads) / ((double)GNA3_NUM_CE / (double)uThreads);
            StreamEff = StreamEff > 1 ? 1 : StreamEff;
            double CNNEff = uThreadEff * KWGEff * StreamEff;

            if (BestEff <= CNNEff)
            {
                BestEff = CNNEff;
                convConfiguration->uT = static_cast<uint8_t>(uThreads);
            }
        }

        return true;
    }

    bool GNA3_GenAdaptHW_UMemBase(convolutional_fused_configuration* const convConfiguration)
    {
        // uArch: UMEM is constructed by 3 Sections (Separate EBBs implemented as RF, 16-Bytes Words), lets refer then as A,B,C
        // UMEM is splited into: UMEM = X + Y + Z = 1/4 + 1/4 + 1/2 (Repectively)
        // Current GNA Desritpor uses 8-bits pointers, meaning: UMEM_Stack = A + B + C <= 4KB = (2^8) * 16B
        // Example for 4 CE(s):
        //                  CE[3] UMEM-Stack:       CE[2] UMEM Stack:       CE[1] UMEM Stack:       CE[0] UMEM Stack:       Row-Index
        //                  +--------------------+  +--------------------+  +--------------------+  +--------------------+     0x0
        //  X = 1/4 UMEM    | X = 1/4 UMEM-Stack |  | X = 1/4 UMEM-Stack |  | X = 1/4 UMEM-Stack |  | X = 1/4 UMEM-Stack |      |
        //                  +--------------------+  +--------------------+  +--------------------+  +--------------------+      |
        //  Y = 1/4 UMEM    | Y = 1/4 UMEM-Stack |  | Y = 1/4 UMEM-Stack |  | Y = 1/4 UMEM-Stack |  | Y = 1/4 UMEM-Stack |      |
        //                  +--------------------+  +--------------------+  +--------------------+  +--------------------+      |
        //                  |                    |  |                    |  |                    |  |                    |      |
        //  Z = 1/4 UMEM    | Z = 1/2 UMEM-Stack |  | Z = 1/2 UMEM-Stack |  | Z = 1/2 UMEM-Stack |  | Z = 1/2 UMEM-Stack |      |
        //                  |                    |  |                    |  |                    |  |                    |      |
        //                  +--------------------+  +--------------------+  +--------------------+  +--------------------+     MAX
        // Any UMEM Section must not hold both KMEM and CMEM. PMEM can be allocated with same section of any other.

        enum SectionMap
        {
            XtoZ, // CMEM,KMEM
            ZtoX, // CMEM,KMEM
            YtoY, // PMEM
            YtoX, // PMEM
            ZtoY, // PMEM
            XtoY, // PMEM
            YtoZ, // PMEM
            ALst  // KMEM
        } KMemSec{ XtoY }, CMemSec{ ZtoX }, PMemSec{YtoZ};

        bool                  AListMemAtmpt = false;
        uint64_t const            YStrtOfst = XSize;
        uint64_t const            YEndOfst = ZSize;
        bool                  CgtK = false; // CMem >= KMem (greater or equal)
        bool                  CltK = false; // CMem >= KMem (less than)

        // Const UMEM Allocation:
        const uint64_t KMemAlloc = convConfiguration->UMemAlloc.KMemAlloc;
        const uint64_t CMemAlloc = convConfiguration->UMemAlloc.CMemAlloc;
        const uint64_t PMemAlloc = convConfiguration->UMemAlloc.PMemAlloc;
        const uint64_t CPMemAlloc = CMemAlloc + PMemAlloc;

        // By default, AdaptHW is not Valid
        convConfiguration->Valid = false;
        convConfiguration->AListMem = false;

        // At first, let's gurantee KMEM+CMEM+PMEM can fit into UMEM:
        if (convConfiguration->UMemAlloc.UMemAlloc <= UMemSize)
        {
            if (GNA3_2DCNN_CNFLCT_EN)
            {
                // Nothing to do, valid config as long as KMEM,CMEM,PMEM can be allocated into UMEM
                // Which is checked in previous 'if' statement
                return true;
            }
            if (convConfiguration->UMemAlloc.CMemAlloc >= convConfiguration->UMemAlloc.KMemAlloc)
            {
                // Mapping KMEM & CMEM (PMEM will be mapped later):
                // If CMEM >= KMEM, allocating CMEM from Z to X, KMEM from X to Z
                CgtK = true;
                CMemSec = ZtoX;
                KMemSec = XtoZ;
                if ((CMemAlloc <= ZSize) && (KMemAlloc <= XSize))
                {
                    PMemSec = YtoY;
                }
                else if ((CMemAlloc <= ZYSize) && (KMemAlloc <= XSize))
                {
                    PMemSec = YtoX;
                }
                else if ((CMemAlloc <= ZSize) && (KMemAlloc <= XYSize))
                {
                    PMemSec = ZtoY;
                }
                else
                {
                    AListMemAtmpt = true;
                }
            }
            else
            { // CMEM < KMEM, allocating CMEM from X downto Z, KMEM from Z upto X
                CltK = true;
                CMemSec = XtoZ;
                KMemSec = ZtoX;
                if ((CMemAlloc <= XSize) && (KMemAlloc <= ZSize)) { PMemSec = YtoY; }
                else if ((CMemAlloc <= XSize) && (KMemAlloc <= ZYSize)) { PMemSec = XtoY; }
                else if ((CMemAlloc <= XYSize) && (KMemAlloc <= ZSize)) { PMemSec = YtoZ; }
                else
                {
                    AListMemAtmpt = true;
                }
            }
        }
        else // KMEM+CMEM+PMEM CANNOT Fit !
        { 
            AListMemAtmpt = true;
        }

        // Stop further mapping/calulation if GNA3_2DCNN_ALIST_EN not supported
        if (AListMemAtmpt && !GNA3_2DCNN_ALIST_EN)
        {
            return false;
        }

        // At this point, evaluate "AListMemAtmpt" ; (If true, there was no successfull mapping w/o AListMEM)
        // Trying using the Active-List Memory as KMEM (AListMem = True) ; Will allocate CMEM onto UMEM (Using X,Y and Z if needed);
        if (AListMemAtmpt)
        {
            convConfiguration->AListMem = true;
            if ((convConfiguration->UMemAlloc.KMemAlloc <= AListMemSize) && (convConfiguration->UMemAlloc.CMemAlloc + convConfiguration->UMemAlloc.PMemAlloc <= UMemSize))
            {
                if (convConfiguration->UMemAlloc.CMemAlloc <= convConfiguration->UMemAlloc.PMemAlloc) { CMemSec = XtoZ; KMemSec = ALst; PMemSec = ZtoX; }
                else /*(AdaptHW->UMemAlloc.CMemAlloc  > AdaptHW->UMemAlloc.PMemAlloc)*/ { CMemSec = ZtoX; KMemSec = ALst; PMemSec = XtoZ; }
            }
            else {
                return false; // No succesfull allocation/mapping into UMEM
            }
        }

        // If we got to this point, there is a Valid allocation/mapping. Let's calculate the UMEM mapping pointers:
        // Allocating CMEM & KMEM Section:
#define GNA3_UMEM_SEC_ALLOC(Sec,AllocSize,Ofst) \
        /* Alocating Active-List */ (Sec == ALst) ? ( 0x0                                         ) : \
        /* Alocating from X-to-Z */ (Sec == XtoZ) ? ( (Ofst) / UMemRowSize                        ) : \
        /* Alocating from Z-to-X */ (Sec == ZtoX) ? ((UMemSize - (Ofst) - AllocSize) / UMemRowSize) : \
        /* ERROR - Not expected  */                 ( 0xDeadBeef                                  )   \

        convConfiguration->CMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(CMemSec, convConfiguration->UMemAlloc.CMemAlloc, 0));
        convConfiguration->KMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(KMemSec, convConfiguration->UMemAlloc.KMemAlloc, 0));

        if (KMemSec == ALst)
        {
            convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(PMemSec, PMemAlloc, 0));
        }
        else if (CgtK) // CMEM >= KMEM !!!
        { 
            if ((0 < KMemAlloc) && (KMemAlloc <= XYSize) == false)  return false;
            if ((0 < CMemAlloc) && (CMemAlloc <= ZYSize) == false)  return false;


            if (PMemAlloc == 0)
            {
                // No PMEM Allocation is needed
            }
            else if (PMemSec == YtoY)
            {
                // KMEM allocated only onto X Segment
                // CMEM allocated only onto Z Segment
                if (PMemAlloc <= YSize)
                { // PMEM allocated only onto Y Segment
                    convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(ZtoX, PMemAlloc, YEndOfst));
                }
                else if (CPMemAlloc <= ZYSize)
                { // PMEM allocated NOT only onto Y Segment !
                 // PMEM is allocating entire Y Segment, and its residue onto Z Segment (shares with CMEM)
                    convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(XtoZ, PMemAlloc, YStrtOfst));
                }
                else
                {
                    // PMEM is allocated in this order:
                    // 1. Whatever space left in Z Segement (Shares Z Segment with CMEM)
                    // 2. Entire Y Segment
                    // 3. Residue in X Segment (Sahres with KMEM, minimized)
                    convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(ZtoX, PMemAlloc, CMemAlloc));
                }
            }
            else if (PMemSec == YtoX)
            {
                // KMEM allocated onto X Segment
                // CMEM allocated onto Z and Y Segments
                // PMEM Starts from Y towards X

                convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(ZtoX, PMemAlloc, CMemAlloc));
            }
            else if (PMemSec == ZtoY)
            {
                // KMEM allocated onto X and Y Segments
                // CMEM allocated onto Z segment
                // PMEM Starts from Z towards Y
                // In this scenario, we will always allocate PMEM at the END of CMEM.
                // Meaning, PMEM will consume all residue of Z Segment (share with CMEM), and Y Segement as needed (Shared with KMEM)

                convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(ZtoX, PMemAlloc, CMemAlloc));
            }
            else 
            {
                return false;
            }
        }
        else if (CltK) // CMEM < KMEM !!!
        { 
            if (PMemAlloc == 0) 
            {
                // No PMEM Allocation is needed
            }
            else if (PMemSec == YtoY)
            {
                // CMEM allocated only onto X Segment
                // KMEM allocated only onto Z Segment
                if (PMemAlloc <= YSize) // PMEM allocated fully only onto Y Segment
                { 
                    convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(XtoZ, PMemAlloc, YStrtOfst));
                }
                else if (CPMemAlloc <= XYSize) // PMEM allocated NOT only onto Y Segment !
                { 
                    // PMEM is allocated onto Y Segment, and its residue onto X Segment (with CMEM)
                    convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(ZtoX, PMemAlloc, YEndOfst));
                }
                else 
                {
                    // PMEM is allocated in this order:
                    // 1. Whatever space left in X Segement (Shares X Segment with CMEM)
                    // 2. Entire Y Segment
                    // 3. Residue in Z Segment (Sahres with KMEM, minimized)
                    convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(XtoZ, PMemAlloc, CMemAlloc));
                }
            }
            else if (PMemSec == XtoY)
            {
                // CMEM allocated onto X Segment
                // KMEM allocated onto Z and Y Segments
                // PMEM Starts from X towards Y

                convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(XtoZ, PMemAlloc, CMemAlloc));
            }
            else if (PMemSec == YtoZ)
            {
                // CMEM allocated onto X and Y Segments
                // KMEM allocated onto Z Segment
                // PMEM Starts from Y towards Z

                convConfiguration->PMemBase = static_cast<uint8_t>(GNA3_UMEM_SEC_ALLOC(XtoZ, PMemAlloc, CMemAlloc));
            }
            else
            {
                return false;
            }
        }

#undef GNA3_UMEM_SEC_ALLOC
        return true;
    }

    bool GNA3_GenAdaptHW_KWG(ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn, const DataMode& outputMode, convolutional_fused_configuration* const convConfiguration, uint32_t inPrec) 
    {
        bool ForceReCalc = false;

        // Temporary storage for Iterations and setting initial values:
        convolutional_fused_configuration AdaptHWRef = *convConfiguration;
        if (!convConfiguration->Valid) 
        {
            *convConfiguration = gna_convolutional_fused_configuration_def;   // Reseting AdptHW Struct
            convConfiguration->uT = AdaptHWRef.uT;      // Restoing the uT
            convConfiguration->Valid = true;               // Initial seting to get into the while loop
        }
        else
        {
            ForceReCalc = true;
        }
        // Extracting CNN Params:

        const DataMode& BPrec = cnnIn->Biases->Mode;
        const auto BType = cnnIn->Biases->BiasMode;
        uint32_t KNum = cnnIn->Filters->Count;
        uint32_t MaxKernels;

        // Calculating Max-Kernels:
        // In case where BIAS is Per Kernel, there is limitation by the BIAS-MEM size
        // Otherewiese, BIAS-Volume is streamed with no limitaion
        if ((BPrec.Value == GNA_DATA_DISABLED) || (KernelBiasModePerStride == BType))
        {

            MaxKernels = KNum;
        }
        else
        {
            // Max-Kernel-Cap is function of the BIAS volume, is case of Bias-Per-Kernel
            // In this case, all BIAS are pre-loaded onto the BIAS-Memory (and is not streamed in)
            uint32_t MaxKernelsCap = uintDivCeil(GNA3_BMEM_SIZE_KB * (float)GNA3_CONST_KB, BPrec.Size);
            MaxKernels = (KNum <= MaxKernelsCap) ? KNum : MaxKernelsCap;
        }

        // Floowing iterative loop, attempts adding additional Kernel to KWG
        // On each iteration, it checks for valid mapping
        while (((convConfiguration->Valid) && (convConfiguration->KWG < MaxKernels)) || ForceReCalc)
        {
            // Adding additional Kernel to Kernel-Working-Group:
            convConfiguration->Valid = false;
            if (ForceReCalc == true)
            {
                ForceReCalc = false;
            }
            else 
            {
                convConfiguration->KWG++;
            }
            // Calculating UMEM Allocation needed:(KMEM + CMEM + PMEM):
            convConfiguration->UMemAlloc.KMemAlloc = GNA3_KMemAllocSize(cnnIn, convConfiguration, inPrec);
            convConfiguration->UMemAlloc.CMemAlloc = GNA3_CMemAllocSize(cnnIn, outputMode, convConfiguration);
            convConfiguration->UMemAlloc.PMemAlloc = GNA3_PMemAllocSize(cnnIn, poolingIn, outputMode, convConfiguration);
            convConfiguration->UMemAlloc.UMemAlloc = convConfiguration->UMemAlloc.KMemAlloc +
                convConfiguration->UMemAlloc.CMemAlloc +
                convConfiguration->UMemAlloc.PMemAlloc;
            // Mapping UMEM Allocation onto Physical-Memories:

            convConfiguration->Valid = GNA3_GenAdaptHW_UMemBase(convConfiguration);

            // Storing last Valid mapping as REF
            if (convConfiguration->Valid)
            {
                AdaptHWRef = *convConfiguration;
            }
        }
        // Checking for Valid Mapping:
        // Checking for Valid Mapping:
        *convConfiguration = AdaptHWRef;
        if (!convConfiguration->Valid)
        {
            return false;
        }

        // Update KWG-Iterations:
        convConfiguration->KWGIter = static_cast<uint8_t>(uintDivCeil(cnnIn->Filters->Count, convConfiguration->KWG));

        return convConfiguration->Valid;
    }

    bool GNA3_PopulateLD(ConvolutionFunction2D const * cnnIn, PoolingFunction2D const * poolingIn, const DataMode& outputMode, convolutional_fused_configuration* const convConfiguration)
    {
        uint32_t KDimW;

        KDimW = cnnIn->Filters->at(GNA_DIM_W);

        // Following Check validates the IFV limitation w.r.t Narrow-Mem
        // It is must that a Kernel-Row can be allocated fully within Narow-Mem
        // Kernel-Row = IFVc * Kw = Kw * IFVc = "KwC"

        uint32_t KwC_B = static_cast<uint32_t>(cnnIn->Input->at(GNA_DIM_D) * cnnIn->Input->Mode.Size * KDimW);
        uint32_t EffNM_B = static_cast<uint32_t>(GNA3_NMEM_SIZE_KB * GNA3_CONST_KB);
        if (GNA3_2DCNN_NMADJ == true)
        {
            EffNM_B -= 16;
        }
        if (KwC_B > EffNM_B)
        {
            return false;
        }


        // Step (1) : Iterative loop, attempts adding additional Kernel to KWG:
        //            We assume at this point a single uThread
        convConfiguration->Valid = false;  // Assuming no Valid Mapping
        convConfiguration->uT = 1;      // Assuming 1 uThread (for now)

        if (!GNA3_GenAdaptHW_KWG(cnnIn, poolingIn, outputMode, convConfiguration, cnnIn->Input->Mode.Size))
        {
            return false;
        }

        // Step (2) : Iterative loop, chooses the right uThread:
        if (!GNA3_GenAdaptHW_uT(cnnIn, convConfiguration))
        {
            return false;
        }

        // Step (3) : Attempting another KWG Iteration, as uThreads has hopefully gone up
        //            In addition his step re-calculates the UMEM Allocation with new Value of uThreads
        if (!GNA3_GenAdaptHW_KWG(cnnIn, poolingIn, outputMode, convConfiguration, cnnIn->Input->Mode.Size))
        {
            return false;
        }

        return true;
    }
}
