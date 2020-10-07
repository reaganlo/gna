set GNA_CMAKE_TOOLSET_NAME=msvc
set GNA_CMAKE_REPO_PATH=..\..
set GNA_CMAKE_TOOLSET="Visual Studio 15 2017 Win64"
set GNA_CMAKE_TOOLSET_CXX="v141"
set GNA_CMAKE_CONFIG=WIN_DEBUG


call .\build-lib.bat || exit /b 666

