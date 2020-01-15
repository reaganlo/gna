// ------------------------------------------------------------------------------------------------
// GNA-3.0 Architecture C-Package Header
// ------------------------------------------------------------------------------------------------
// Change-Bar:
// 1. Adding 1D-CNN Support
// 2. Adapt-HW algorythem match GNA-3.0 Rev 0.81 HAS
// 3. Some minor changes for PSV integration
// [+] 1. Fixed Narrow-Mem Limitation Calulcation (Was missung the IFc Prec)
// [+] 2. Update Latest 2D-CNN/1D-CNN uArch Limitation Narrow-Mem == 2KB --> Narrow-Mem == 2KB-16B
// [+] 3. Update PopulateLD to overwrite AdaptHW.Valid = false, in case Op is not supported
// [+] 4. Remove 'MAC_Prec' from Layer-Descritors (1DCNN/2DCNN)
// [+] 5. Effective-PMEM Precision - PMEM Element Size is Similar to CNV Element Size (Previous Element-Size was eq to ACTx)
// [+] 6. 2D-CNN Performace ; TODO/FIXME Zero-Padding
// [+] 7. AFFINE Performace
// [+] 8. DE-INTERLEAVE Performace
// [+] 9. HSD:1409847156  | [GNA3.0][Arch CPkg] For 1B Kernel / 2B Input for 2DCNN C - Package need to handle the KWGIteration considering that only 1 / 2 the Widemem is available due to padding of 1B -> 2B
// [+] 10.HSD:14010047816 | [GNA3.0][Arch CPkg] PMEM Allocation for SUM Pooling
// [+] 11.HSD:1409847274  | [GNA3.0][Arch CPkg] Need to update Narrowmem uArch limitation in HAS and CPkg {NMEM = 2KB - 16}
// [+] 12.HSD:14010296027 | [GNA3.0][Arch CPkg] UMem Allocation when adding UThread
// [+] 13.HSD:14010385679 | [GNA3.0][Arch CPkg][HAS] PMEM allocation needs to be incresed by 1 in certain non-aligned configurations
// [+] 14.HSD:14010525820 | [GNA3.0][HAS] 2D CNN Narrowmem limitation needs to be updated for 1B input 2B Kernel precision
// [+] 15.                | Checks for KNum Constraints w.r.t ACTx (1,2,4n...)
// [+] 16.                | Checks for Pooling Constraints (Pool-Window <= CNV)
// [+] 17.                | [GNA3.0][Arch CPkg] Need to update Narrowmem uArch limitation in HAS and CPkg {NMEM = 2KB - 16 - 128}
// [+] 18.HSD:NONE        | Update CNNs Bias-Mode (BMODE) Enumeration values to match HAS
// [+] 19.HSD:14010657629 | [GNA3.0] [Arch CPkg] Need to Update MAX KWGSize limitation for some unaligned corner cases
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// WIP - WISHLIST - TODOs
// 1. Performance with Sub-System Aware: Outstanding
// 2. CVU: TODO/FIXME Zero-Padding
//         Data-Forwarding, reduced CMEM aceesses (Not simulate)
// 3. AFU: Fast Relu
// 4. OSU: Perf (OCP vs IOSF) - Currently no diff, and no write bubbles
// 5. 1D-CNN
// 6. xNNs
// ------------------------------------------------------------------------------------------------

#ifndef GNA_ArchCPkg 
#define GNA_ArchCPkg

extern "C" { // Will avoid 'cpp' renemaing of functions ; Requierd by SVTB (ENG Team)

#include <stdint.h>  // Usage of: uintx_t
#include <stdbool.h> // Usage of: bool

    // ------------------------------------------------------------------------------------------------
    // Typedef(s) & Defines:
    // ------------------------------------------------------------------------------------------------
    typedef uint32_t uint;     // unsigned integer 4B
    typedef uint64_t ullg;     // unsigned integer 8B
    typedef uint8_t  uint4_t;  // 4-bits
    
    #define GNA3_PWLF_ReLU     0x81      // Used for Perfmance
    #define GNA3_PWLF_ClmpRELU 0x82      // Used for Perfmance

    // ------------------------------------------------------------------------------------------------
    // Global(s) & GNA Configuration:
    // ------------------------------------------------------------------------------------------------
    typedef struct GNA3_Config {    // GNA-3.0 Config Knobs (Order Matters!):
        // Arch Configuraion knobs:
        uint GNA3_GEN;              // GNA Generation ; GNA-3.0 --> 30
        uint GNA3_NUM_CE;           // Number of CE(s) 
        uint GNA3_UMEM_SIZE_KB;     // Total Unified-Memory Size in KB
        // Internal uArch Features:
        bool GNA3_2DCNN_ALIST_EN;   // Enables Active-List as memory for 2D-CNN
        bool GNA3_2DCNN_CNFLCT_EN;  // Allows conflicts in UMEM between KMEM and CMEM
        bool GNA3_2DCNN_PLUPACK_EN; // Allows conflicts in UMEM between KMEM and CMEM
        bool GNA3_2DCNN_C1MEM_EN;   // Allows CMEM_Elements = CNVw * 1 ; instead of CMEM_Elements = CNVw * Kh ; Only if (Kh * IFVw * IFVc) feets in Narrow-MEM
    } GNA3_Config_t;
    typedef struct GNA3_SysConfig { // GNA-3.0 Sus-System Config Knobs:
        uint  SYS_FREQ_MHz;         // System/GNA Frequancy [MHz]
        float SYS_IF_BW_BpC;        // Interface Bandwith (DMA) [Bytes/Cycle]
        uint  SYS_MEM_LAT_nSec;     // External-Memory (DDR/SRAM) Latency in [nSec]
        uint  SYS_POUTS_CNT;        // Number of Posted (Writes) Outstanding(s) [Count]
        uint  SYS_NPOUTS_CNT;       // Number of Non-Posted (Read+Writes) Outstanding(s) [Count]
    } GNA3_SysConfig_t;

    // ------------------------------------------------------------------------------------------------
    // Stractures/Enums:
    // ------------------------------------------------------------------------------------------------

    // Premitives/Basline Stractures:
    typedef enum   GNA3_Prec {
        // GNA Precision Supported: (Note INT32 only supported for BIAS)
        // Note: Enum values must match the number of bytes it represents !
        GNA3_DIS   = 0,
        // Data-Type Precision:
        GNA3_INT8  = 1,
        GNA3_INT16 = 2,
        GNA3_INT32 = 4,
        // MAC Precision: Number of bytes in CMEM for allocating a single Accumolator
        GNA3_MAC4B = 4,
        GNA3_MAC8B = 8,
        // RICH Format (xNN Only, with INT8 Weights)
        GNA3_RICH  = 8,
    } GNA3_Prec_t;
    typedef enum   GNA3_BiasType { // GNA Bias Type:
        GNA3_BIASperSTRIDE = 0,    // BIASperSTRIDE - Single BIAS term for each Output Stride (After Convolution & Pooling)
        GNA3_BIASperKERNEL = 1     // BIASperKERNAL - Single BIAS term for associaited with each Kernel
    } GNA3_BiasType_t;
    typedef enum   GNA3_PoolType { // Pooling Type
        GNA3_POOL_DIS = 0,         // Disabled
        GNA3_POOL_MAX = 1,         // MAX Pooling
        GNA3_POOL_SUM = 2          // SUM Pooling
    } GNA3_PoolType_t;
    typedef struct GNA3_Tensor {   // NHWC
        uint16_t    N;             // Number of 3D-Tensors (Used as 4th diemension)
        uint16_t    H;             // Height (Horizontal) Dimension
        uint16_t    W;             // Width (Vertical) Dimension
        uint16_t    C;             // Channel (Depth) Dimension
        GNA3_Prec_t Prec;          // Precision
    } GNA3_Tensor_t;
    typedef enum   GNA3_TensorType {
        GNA3_IFV   = 0,
        GNA3_KRV   = 1,
        GNA3_BIASV = 2,
        GNA3_OUTV  = 3,
    } GNA3_TensorType_t;
    typedef struct GNA3_Plain {
        uint16_t H;
        uint16_t W;
    } GNA3_Plain_t;
    typedef struct GNA3_UMemAlloc {
        uint  KMemAlloc; // KMEM Allocation in Bytes (Gross)
        uint  CMemAlloc; // CMEM Allocation in Bytes (Gross)
        uint  PMemAlloc; // PMEM Allocation in Bytes (Gross)
        uint  UMemAlloc; // UMEM Allocation in Bytes (Gross) ; Total of K+C+P
        float PrbKPCflt; // Probablity KMem-PMem Conflict
        float PrbCPCflt; // Probablity CMem-PMem Conflict
    } GNA3_UMemAlloc_t;

    // Specific (Layers) Stractures:
    typedef struct GNA3_1DCNN {
        // Kernel Parameters:
        uint16_t        NConvFilters;           // Number of Kernels        {[1..2048] ; 0 ilegal}
        uint16_t        NConvFilterElements;    // Kernel Size in Elements  {[1..1024,2048] ; 0 ilegal}
        GNA3_Prec_t     KPrec;                  // Kernel Precision
        // Convolution Parameters:
        uint16_t        InputConvStride;        // Convolution Stride(s)    {[1..1024,2048] ; 0 ilegal}
        // Pooling Parameters:
        GNA3_PoolType_t PType;                  // Pooling Type (DIS/MAX/SUM)
        uint8_t         PWin;                   // Pooling Window           {[0..255] ; 0 means 256}
        uint8_t         PStr;                   // Pooling Stride           {[0..255] ; 0 means 256}
        // Bias:
        GNA3_Prec_t     BPrec;                  // Bias Precision
        GNA3_BiasType_t BType;                  // Bias Type (per Kernel/Stride)
        // Activation (ACTx):
        GNA3_Prec_t     ACTx;                   // Activation Precision PWLF (Output-Volume)
        uint16_t        NSegs;                  // Number of Segments [0x01..0x80 ; 0x81 = ReLU ; 0x82 = Clamp ReLU]
    } GNA3_1DCNN_t;
    typedef struct GNA3_2DCNN {
        // Kernel Parameters:
        uint16_t        KNum;                   // Number of Kernels (N diemnsion in NHWC)
        GNA3_Plain_t    KDim;                   // Kernel (upto 256 on each dimenssion)
        GNA3_Prec_t     KPrec;                  // Kernel Precision
        // Convolution Parameters:
        GNA3_Plain_t    CStr;                   // Convolution Stride(s)
        GNA3_Plain_t    CZPad;                  // Convolution Zero-Padding
        // Pooling Parameters:                    
        GNA3_PoolType_t PType;                  // Pooling Type (DIS/MAX/SUM)
        GNA3_Plain_t    PWin;                   // Pooling Window
        GNA3_Plain_t    PStr;                   // Pooling Stride
        // Bias:                                
        GNA3_Prec_t     BPrec;                  // Bias Precision
        GNA3_BiasType_t BType;                  // Bias Type (per Kernel/Stride)
        // Activation (ACTx):                    
        GNA3_Prec_t     ACTx;                   // Activation Precision PWLF (Output-Volume)
        uint16_t        NSegs;                  // Number of Segments [0x01..0x80 ; 0x81 = ReLU ; 0x82 = Clamp ReLU]
    } GNA3_2DCNN_t;
    typedef struct GNA3_AFFINE {
        // Inputs/Outputs
        uint16_t        NInputElements;         // Number of Input Elements
        uint16_t        NOutputElements;        // Number of Output Elements
        uint8_t         Grouping;               // Grouping/Batching Value
        GNA3_Prec_t     InPrc;                  // Input-Precision
        GNA3_Prec_t     WKPrc;                  // Weights-Precision
        // Bias:                                
        GNA3_Prec_t     BPrec;                  // Bias Precision
        // Activation (ACTx):                    
        GNA3_Prec_t     ACTx;                   // Activation Precision PWLF (Output-Volume)
        uint16_t        NSegs;                  // Number of Segments [0x01..0x80 ; 0x81 = ReLU ; 0x82 = Clamp ReLU]
    } GNA3_AFFINE_t;
    typedef struct GNA3_DEINTRLV {
        // Inputs/Outputs
        uint16_t        NInputElements;         // Number of Input Elements
        uint8_t         Grouping;               // Grouping/Batching Value
        GNA3_Prec_t     InPrc;                  // Input-Precision
    } GNA3_DEINTRLV_t; 
    // Generic Layer-Structure - UNION
    typedef union  GNA3_OpStruct {
        GNA3_AFFINE_t   GNA3_OP_AFFINE;
        GNA3_DEINTRLV_t GNA3_OP_DEINTRLV;
        GNA3_1DCNN_t    GNA3_OP_1DCNN;
        GNA3_2DCNN_t    GNA3_OP_2DCNNc;
    } GNA3_OpStruct_t;
    
    // Generic Layer Descriptor:
    typedef enum   GNA3_Ops {
        GNA3_OP_AFFINE        = 0x00,
     // GNA3_OP_AFFINE_AL     = 0x01,
     // GNA3_OP_DIAGONAL      = 0x02,
     // GNA3_OP_RNN           = 0x04,
        GNA3_OP_1DCNN         = 0x08,
     // GNA3_OP_AFFINE_MBG    = 0x09,
        GNA3_OP_DEINTRLV      = 0x10,
     // GNA3_OP_INTERLEAVE    = 0x11,
     // GNA3_OP_COPY          = 0x12,
     // GNA3_OP_GMM           = 0x20,
     // GNA3_OP_GMM_AL        = 0x21,
        GNA3_OP_2DCNNc        = 0x30, // 2D-CNN Fused Layer
     // GNA3_OP_2DCNNp        = 0x31, // 2D-CNN Pooling Only Layer
     // GNA3_OP_2DCNNa        = 0x32, // 2D-CNN Addition Layer
     // GNA3_OP_2DCNNv        = 0x33, // 2D-CNN Conversion Layer
    } GNA3_Ops_t;    
    typedef struct GNA3_AdaptHW {
        bool             Valid;     // Indiacates Valid AdaptHW Configuration
        uint             DPWidth;   // Datapath Width (1=INT16,2=INT8)
        uint16_t         KWG;       // GNA-3.0 HAS : Kernel-Working-Group (Number of Kernels in IFV Iteration)
        uint16_t         KWGIter;   // GNA-3.0 HAS : Kernel-Working-Group Iterations
        uint8_t          uT;        // GNA-3.0 HAS : Micro-Threads (4-bits)
        uint8_t          KMemBase;  // GNA-3.0 HAS : GNA Descriptor
        uint8_t          CMemBase;  // GNA-3.0 HAS : GNA Descriptor
        uint8_t          PMemBase;  // GNA-3.0 HAS : GNA Descriptor
        bool             AListMem;  // TODO
        GNA3_UMemAlloc_t UMemAlloc; // UMEM Allocation MetaData
    } GNA3_AdaptHW_t;
    typedef struct GNA3_LyrDesc {
        GNA3_Tensor_t   IFV;        // Input-Feature-Volume {4D+Precision}
        GNA3_Tensor_t   KRV;        // Kernel-Volume(CNN)/Weights(xNN) {4D+Precision}
        GNA3_Tensor_t   BIASV;      // Bias-Volume {4D+Precision}
        GNA3_Tensor_t   OUTV;       // Output-Volume {4D+Precision}
        GNA3_Ops_t      Op;         // Operation Type (Layer)
        GNA3_OpStruct_t OpStruct;   // Operation Struct (Union)
        GNA3_AdaptHW_t  AdaptHW;    // [uArch] Adaptive-HW Parameters
        void*           ExtPerf;    // Performance Extension
    } GNA3_LyrDesc_t;
             
    // PnP Stracures: ---------------------------------------------------------------------------------
 

    // ------------------------------------------------------------------------------------------------
    // [ARCH] API:
    // ------------------------------------------------------------------------------------------------
    
    GNA3_LyrDesc_t* GNA3_NewLD();                                                          // Layer-Descriptor Constractor (Alloc & Init)
    void            GNA3_FreeLD(GNA3_LyrDesc_t* const LD);                                 // Layer-Descriptor Distractor (Free)
        
    bool GNA3_PopLD(GNA3_LyrDesc_t* const LD);                                             // Populate GNA Layer-Desciptor

    uint GNA3_GetVolElms(const GNA3_LyrDesc_t* const LD, const GNA3_TensorType_t VolType); // Retrives Number of Elements in Volume
    uint GNA3_GetVolSize(const GNA3_LyrDesc_t* const LD, const GNA3_TensorType_t VolType); // Retrives Volume size in Bytes

    bool GNA3_SetConfig(const GNA3_Config_t* const Config);                                // Configures GNA-HW
    bool GNA3_GetConfig(      GNA3_Config_t* const Config);                                // Retrives current GNA-HW Configuration

    // ------------------------------------------------------------------------------------------------
    // [uARCH] API:
    // ------------------------------------------------------------------------------------------------

    GNA3_Tensor_t GNA3_GetCNV(const GNA3_LyrDesc_t* const LD); // Computes GNA-3.0 Convolution - Volume Tensor Dimensions.
    GNA3_Tensor_t GNA3_GetPLV(const GNA3_LyrDesc_t* const LD); // Computes GNA-3.0 Pooling-Volume Tensor Dimensions.

    uint GNA3_GetKnMemElmts(const GNA3_LyrDesc_t* const LD);   // Computes: Number of Elements (not Size!) of a SINGLE Kernel (Kn) allocation.
    uint GNA3_GetCnMemElmts(const GNA3_LyrDesc_t* const LD);   // Computes: Number of Elements (not Size!) of a SINGLE CMEM (Cn) allocation.
    uint GNA3_GetPnMemElmts(const GNA3_LyrDesc_t* const LD);   // Computes: Number of Elements (not Size!) of a SINGLE PMEM (Pn) allocation.

} // extern C
#endif // GNA_ArchCPkg
