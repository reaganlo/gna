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

/*++

Module Name:

    Trace.h

Abstract:

    Header file for the debug tracing related function defintions and macros.

Environment:

    Kernel mode

--*/

//
// Define the tracing GUID and flags.
//

#define WPP_CONTROL_GUIDS                                           \
    WPP_DEFINE_CONTROL_GUID(                                        \
        GnaDrvTraceGuid, (905E9841,7270,48B6,9210,0104A52CD07F),    \
                                                                    \
        WPP_DEFINE_BIT(T_DRV)                                       \
        WPP_DEFINE_BIT(T_QUE)                                       \
        WPP_DEFINE_BIT(T_INIT)                                      \
        WPP_DEFINE_BIT(T_IOCTL)                                     \
        WPP_DEFINE_BIT(T_REG)                                       \
        WPP_DEFINE_BIT(T_MEM)                                       \
        WPP_DEFINE_BIT(T_ENT)                                       \
        WPP_DEFINE_BIT(T_EXIT)                                      \
        )                             

#define TLF TRACE_LEVEL_FATAL
#define TLE TRACE_LEVEL_ERROR
#define TLW TRACE_LEVEL_WARNING
#define TLI TRACE_LEVEL_INFORMATION
#define TLV TRACE_LEVEL_VERBOSE

//
// Basic trace function
//
// begin_wpp config
// FUNC Trace(LEVEL, FLAGS, MSG, ...);
// end_wpp
#define WPP_FLAG_LEVEL_LOGGER(  flag, level)  WPP_LEVEL_LOGGER( flag )
#define WPP_FLAG_LEVEL_ENABLED( flag, level) (WPP_LEVEL_ENABLED(flag ) && WPP_CONTROL(WPP_BIT_ ## flag ).Level >= level)
#define WPP_LEVEL_FLAGS_LOGGER( lvl,  flags)  WPP_LEVEL_LOGGER( flags)
#define WPP_LEVEL_FLAGS_ENABLED(lvl,  flags) (WPP_LEVEL_ENABLED(flags) && WPP_CONTROL(WPP_BIT_ ## flags).Level >= lvl)


// 
// Function entry and exit tracing macros
//

// begin_wpp config
// FUNC TraceEntry(LEVEL, FLAGS);
// USESUFFIX(TraceEntry, "%!FUNC! Entry");
// end_wpp

// begin_wpp config
// FUNC TraceFail(LEVEL, FLAGS, EXP);
// USESUFFIX(TraceFail, "%!FUNC! Failed with status==%!STATUS!",EXP);
// end_wpp
#define WPP_LEVEL_FLAGS_EXP_ENABLED(l, f, status) WPP_LEVEL_FLAGS_ENABLED(l, f)
#define WPP_LEVEL_FLAGS_EXP_LOGGER(l, f, status)  WPP_LEVEL_LOGGER(f)


// begin_wpp config
// FUNC TraceFailMsg(LEVEL, FLAGS, FN, STS);
// USESUFFIX(TraceFailMsg, "%!FUNC!: %s failed with status==%!STATUS!", FN, STS);
// end_wpp
#define WPP_LEVEL_FLAGS_FN_STS_ENABLED(l, f, msg, sts) WPP_LEVEL_FLAGS_ENABLED(l, f)
#define WPP_LEVEL_FLAGS_FN_STS_LOGGER(l, f, msg, sts)  WPP_LEVEL_LOGGER(f)

//
//begin_wpp config
//FUNC TraceReturn(LEVEL, FLAGS, EXP2);
//USESUFFIX (TraceReturn, "%!FUNC! Failed with status==%!STATUS!",EXP2);
//end_wpp
#define WPP_LEVEL_FLAGS_EXP2_PRE(l, f, status)  {if (STATUS_SUCCESS != status) {
#define WPP_LEVEL_FLAGS_EXP2_POST(l, f, status) ;}}
#define WPP_LEVEL_FLAGS_EXP2_ENABLED(l, f, status) WPP_LEVEL_FLAGS_ENABLED(l, f)
#define WPP_LEVEL_FLAGS_EXP2_LOGGER(l, f, status)  WPP_LEVEL_LOGGER(f)
