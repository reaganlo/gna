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

#if !defined(_SCORE_PROCESSOR_H)
#define _SCORE_PROCESSOR_H

#include "Driver.h"

/**
 * Handles score request dispatching on write event
 */
EVT_WDF_IO_QUEUE_IO_WRITE
ScoreSubmitEvnt;

/**
 * Completes processing and resets state
 *
 * @devCtx              Device context
 * @status              status of completion
 * @request             optional request to complete if not already set or NULL if set
 * @hwUnmap             if perform hw memory mapping clean
 * @appCancel           app context for which request is completed if is current
 */
VOID
ScoreComplete(
    _In_    PDEV_CTX    devCtx,
    _In_    NTSTATUS    status,
    _In_opt_ WDFREQUEST request,
    _In_    BOOLEAN     hwUnmap,
    _In_opt_ PAPP_CTX   appCancel);

/**
 * Retrieves and cancels all request in queue
 *
 * @devCtx              device context
 * @queue               queue to empty
 * @app                 application file object or NULL for all apps
 */
VOID
ScoreCancelReqByApp(
    _In_    WDFQUEUE    queue,
    _In_    WDFFILEOBJECT app);

/**
 * Scoring ISR interrupt event handler
 */
EVT_WDF_INTERRUPT_ISR
InterruptIsrEvnt;

/**
 * Scoring DPC interrupt event handler
 */
EVT_WDF_INTERRUPT_DPC
InterruptDpcEvnt;

/**
 * Score execution timeout event handler
 */
EVT_WDF_TIMER
ScoreTimeoutEvnt;

/**
 * When queue is idle (no more requests to process) puts hw into low power state
 *
 * @devCtx              context of device
 */
VOID
ScoreProcessorSleep(
    _In_    PDEV_CTX    devCtx);

/**
 * Puts hw into power-on state
 *
 * @devCtx              context of device
 */
VOID
ScoreProcessorWakeup(
    _In_    PDEV_CTX    devCtx);


/**
* Releases application requests from queue and performs unmap
*
* @devCtx              context of device
* @unmapReq            unmap request to process
*/
VOID
ScoreDeferredUnmap(
    _In_    PDEV_CTX    devCtx,
    _In_    WDFREQUEST  unmapReq);

#endif // _SCORE_PROCESSOR_H
