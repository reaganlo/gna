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
#include "SwHw.h"

namespace GNA
{

class WinHandle
{
public:
    WinHandle() :h_(INVALID_HANDLE_VALUE) {}
    /**
     * Deleted functions to prevent from being defined or called
     * @see: https://msdn.microsoft.com/en-us/library/dn457344.aspx
     */
    WinHandle(const WinHandle &) = delete;
    WinHandle& operator=(const WinHandle&) = delete;

    explicit WinHandle(HANDLE h) :h_(h) {}
    ~WinHandle() { Close(); }

    void Set(HANDLE h) { Close(); h_ = h; }
    void Close() { if (INVALID_HANDLE_VALUE != h_) { CloseHandle(h_); h_ = INVALID_HANDLE_VALUE; } }
    operator HANDLE() { return h_; }
private:
    HANDLE h_;
};

class IoctlSender
{
protected:
    IoctlSender();

    void Open(const GUID& guid);

    void Close();

    status_t IoctlSend(DWORD code,
        LPVOID inbuf,
        DWORD inlen,
        LPVOID outbuf,
        DWORD outlen,
        BOOLEAN async = FALSE);

    status_t Submit(
        LPVOID      inbuf,
        DWORD       inlen,
        prof_tsc_t* ioctlSubmit,
        io_handle_t*  handle);

    status_t IoctlWait(
        LPOVERLAPPED ioctl,
        DWORD        timeout);

    DWORD Cancel();

    WinHandle h_;
    WinHandle evt_;
    OVERLAPPED overlapped_;

    /**
     * Prints WinAPI last error info to console
     *
     * @error error code from GetLastError()
     */
    inline static void printLastError(DWORD error)
    {
        LPVOID lpMsgBuf;
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            error,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0,
            nullptr);

        // Display the error message
        if (nullptr != lpMsgBuf)
        {
            wprintf(L"%s\n", (wchar_t*)lpMsgBuf);
            LocalFree(lpMsgBuf);
        }
    }

private:
    IoctlSender(const IoctlSender &) = delete;
    IoctlSender& operator=(const IoctlSender&) = delete;
};

}
