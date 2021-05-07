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


echo Copying samples to package
xcopy /E /Q /S /Y ..\samples ..\bin\samples\ || exit /b 666

mkdir .\samples\

cd .\samples\ || exit /b 666

cmake -G "Visual Studio 15 2017 Win64"  ..\..\bin\samples\ || exit /b 666
echo Created samples project

cmake --build . || exit /b 666
echo Built samples project

echo Running samples project
..\..\bin\samples\bin\sample01\Debug\sample01.exe

echo Cleaning samples project
cd .. || exit /b 666
:: on QB no need to clean workspace, local can be reused
::rmdir /S /Q .\samples\  || exit /b 666
del /F /S /Q ..\bin\samples\bin\ 1>nul
rmdir /S /Q ..\bin\samples\bin\  || exit /b 666
