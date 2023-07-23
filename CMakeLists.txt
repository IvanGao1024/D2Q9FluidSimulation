cmake_minimum_required(VERSION 3.16)

project(FastFluidDynamicSimulationForGames LANGUAGES CXX VERSION 1.0)

### Version
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
        
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### Output Setting
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/")
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib/")
set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/../deploy/")
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

### Output compile commands for clang-dity to locate headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

### Include QT header and tool chain
set(CMAKE_PREFIX_PATH "${QT_LIB_PATH}/cmake/")
set(CMAKE_TOOLCHAIN_FILE "${QT_LIB_PATH}/cmake/Qt6/qt.toolchain.cmake")

add_subdirectory(src)