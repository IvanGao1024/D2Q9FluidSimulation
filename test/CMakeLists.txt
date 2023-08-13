# Find Package
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Find OpenXX
find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

# Find QT
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

## Executable Test
add_executable(MainTests
    # core/CartesianMatrixTest.cpp
    # core/LatticeBoltzmannMethodD2Q9Test.cpp
    # core/OpenCLMainTest.cpp
    )
target_link_libraries(MainTests GTest::gtest_main)
target_link_libraries(MainTests GTest::gtest)
target_link_libraries(MainTests GTest::gmock)
target_link_libraries(MainTests OpenMP::OpenMP_CXX)
target_link_libraries(MainTests OpenCL::OpenCL)
target_link_libraries(MainTests Qt6::Core)
target_link_libraries(MainTests Qt6::Gui)
target_link_libraries(MainTests Qt6::Widgets)
target_compile_options(MainTests PRIVATE --coverage)
target_link_options(MainTests PRIVATE --coverage)

# Executable Benchmark
add_executable(MainBenchmarks
    # core/CartesianMatrixBenchmark.cpp
    # core/LatticeBoltzmannMethodD2Q9Benchmark.cpp
    core/OpenCLMainBenchmark.cpp)
target_link_libraries(MainBenchmarks benchmark::benchmark_main)
target_link_libraries(MainBenchmarks benchmark::benchmark)
target_link_libraries(MainBenchmarks OpenMP::OpenMP_CXX)
target_link_libraries(MainBenchmarks OpenCL::OpenCL)
target_link_libraries(MainBenchmarks Qt6::Core)
target_link_libraries(MainBenchmarks Qt6::Gui)
target_link_libraries(MainBenchmarks Qt6::Widgets)