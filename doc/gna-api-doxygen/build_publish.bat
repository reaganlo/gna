@echo off

:: Copyright 2020 Intel Corporation.

:: This software and the related documents are Intel copyrighted materials,
:: and your use of them is governed by the express license under which they
:: were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
:: Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
:: otherwise, you may not use, modify, copy, publish, distribute, disclose or
:: transmit this software or the related documents without Intel's prior
:: written permission.
:: This software and the related documents are provided as is, with no
:: express or implied warranties, other than those that are expressly
:: stated in the License.

echo Building and publish  GNA API Doxygen documentation

echo PROJECT_NUMBER=%GNA_LIBRARY_VERSION% >> Doxyfile || exit /b 666
doxygen || exit /b 666
