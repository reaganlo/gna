// ------------------------------------------------------------------------------------------------
// GNA-3.0 Architecture C-Package Header - Performace Extension
// ------------------------------------------------------------------------------------------------

#ifndef GNA_ArchCPkg_ExtPerf 
#define GNA_ArchCPkg_ExtPerf

#include "GNA_ArchCPkg.h"
#include <stdint.h>  // Usage of: uintx_t
#include <stdbool.h> // Usage of: bool

extern "C" { // Will avoid 'cpp' renemaing of functions ; Requierd by SVTB (ENG Team)
    
    // ------------------------------------------------------------------------------------------------
    // Stractures/Enums:
    // ------------------------------------------------------------------------------------------------
    static const int GNA3_CmptUnitsNum = 4; // CVU[1], AFU[2], PLU[3], OSU[4]
    
    typedef enum   GNA3_CmptUnitsMap {
        CVU = 0,    // First Unit in Pipe must have enum Value of '0' (Zero)
        AFU = 1,
        PLU = 2,
        OSU = 3,
        N_A         // N_A ,ust be last Index
    } GNA3_CmptUnitsMap_t;
    typedef struct GNA3_UnitPerf_BW {
        float Rd;           // [B/Cyc]
        float Wr;           // [B/Cyc]
    } GNA3_UnitPerf_BW_t;
    typedef struct GNA3_UnitPerf {
        ullg  CycStart;     // Unit Start Cycle Number
        ullg  CycEnd;       // Unit End Cycle Number
        float Util;         // Precentage
    } GNA3_UnitPerf_t;
    typedef struct GNA3_Perf {
        GNA3_UnitPerf_t    PreLoad;
        GNA3_UnitPerf_t    Compute[GNA3_CmptUnitsNum];
        ullg               TtlCycs;
        ullg               PreloadCycs;
        ullg               ComputeCycs;
        ullg               CfltCycs;
        GNA3_UnitPerf_BW_t PreloadBW;    // Bandwith
        GNA3_UnitPerf_BW_t ComputeBW;    // Bandwith
    } GNA3_Perf_t;

    // ------------------------------------------------------------------------------------------------
    // [ARCH] API:
    // ------------------------------------------------------------------------------------------------
    bool GNA3_PopPF(GNA3_LyrDesc_t* const LD); // Populate Performance

} // extern "C"
#endif
