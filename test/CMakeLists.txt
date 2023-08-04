# Find OpenMP
find_package(OpenMP REQUIRED)

# Find QT
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Find GTest
find_package(GTest REQUIRED)
add_executable(MainTests
    core/CartesianMatrixTest.cpp
    core/LatticeBoltzmannMethodD2Q9Test.cpp)
target_link_libraries(MainTests GTest::gtest_main)
target_link_libraries(MainTests GTest::gtest)
target_link_libraries(MainTests GTest::gmock)
target_link_libraries(MainTests OpenMP::OpenMP_CXX)
target_compile_options(MainTests PRIVATE --coverage)
target_link_options(MainTests PRIVATE --coverage)
target_link_libraries(MainTests Qt6::Core)
target_link_libraries(MainTests Qt6::Gui)
target_link_libraries(MainTests Qt6::Widgets)

# Find benchmark
find_package(benchmark REQUIRED)
add_executable(MainBenchmarks
    core/CartesianMatrixBenchmark.cpp
    core/LatticeBoltzmannMethodD2Q9Benchmark.cpp)
target_link_libraries(MainBenchmarks benchmark::benchmark_main)
target_link_libraries(MainBenchmarks benchmark::benchmark)
target_link_libraries(MainBenchmarks OpenMP::OpenMP_CXX)
target_link_libraries(MainBenchmarks Qt6::Core)
target_link_libraries(MainBenchmarks Qt6::Gui)
target_link_libraries(MainBenchmarks Qt6::Widgets)