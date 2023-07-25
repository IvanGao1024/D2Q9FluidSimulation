# Find OpenMP
find_package(OpenMP REQUIRED)

# Find QT
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    main.cpp
    LatticeMainWindow.h
    LatticeMainWindow.cpp
    core/Matrix.hpp
    core/LatticeBoltzmannMethod.h
    core/LatticeBoltzmannMethod.cpp)

set(PROJECT_EXECUTABLE_NAME ${PROJECT_NAME})
add_executable(${PROJECT_EXECUTABLE_NAME} ${PROJECT_SOURCES})
install(TARGETS ${PROJECT_EXECUTABLE_NAME} DESTINATION "deploy")

# Link QT lib
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Qt6::Core)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Qt6::Gui)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Qt6::Widgets)

# Link OpenMP lib
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PUBLIC OpenMP::OpenMP_CXX)

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