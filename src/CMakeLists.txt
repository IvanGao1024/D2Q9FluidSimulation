# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

# Find QT
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    main.cpp
    LatticeMainWindow.h
    LatticeMainWindow.cpp
    core/Matrix.hpp
    core/LatticeBoltzmannMethodD2Q9.h
    core/LatticeBoltzmannMethodD2Q9.cpp
    core/OpenCLMain.hpp)

set(PROJECT_EXECUTABLE_NAME ${PROJECT_NAME})
add_executable(${PROJECT_EXECUTABLE_NAME} ${PROJECT_SOURCES})
install(TARGETS ${PROJECT_EXECUTABLE_NAME} DESTINATION "deploy")

# Link OpenMP lib
target_link_libraries(${PROJECT_EXECUTABLE_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} OpenCL::OpenCL)

target_include_directories(${PROJECT_EXECUTABLE_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_EXECUTABLE_NAME} ${OpenCL_LIBRARIES})

# Link QT lib
target_link_libraries(${PROJECT_EXECUTABLE_NAME} Qt6::Core)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} Qt6::Gui)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} Qt6::Widgets)

# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
#   DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# )


# # FetchContent_Declare(
# #   googlebenchmark
# #   URL https://github.com/google/benchmark/archive/refs/tags/v1.8.2.zip
# #   DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# # )

# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# # MakeAvailable
# FetchContent_MakeAvailable(googletest)
# # FetchContent_MakeAvailable(googlebenchmark)

# Add tests