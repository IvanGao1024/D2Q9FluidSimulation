#!/bin/bash
# environment variable

# QT_LIB_PATH=/6.2.0/gcc_64/lib
QT_LIB_PATH=/home/ivan/QT6.5.1/6.5.1/gcc_64/lib

# default value
Command=style
Platform=linux

# shell starts here
for i in "$@"
do
case $i in
    -command=style)
    Command=style
    shift # past argument=value
    ;;
    -command=doxygen)
    Command=doxygen
    shift # past argument=value
    ;;
    -command=build)
    Command=build
    shift # past argument=value
    ;;
    -command=test)
    Command=test
    shift # past argument=value
    ;;
    -command=run)
    Command=run
    shift # past argument=value
    ;;
    -platform=linux)
    Platform=linux
    shift # past argument=value
    ;;
    -platform=windows)
    Platform=windows
    shift # past argument=value
    ;;
    -platform=mac)
    Platform=mac
    shift # past argument=value
    ;;
    *)
    echo "Invalid input specified: $i"
    ;;
esac
done

case $Command in
    style) # build style through clang-format and clang-tidy
        find src/ -path src/thirdparty -prune -o -name '*.hpp' -exec clang-format -style=file --verbose -i "{}" \; -print
        find src/ -path src/thirdparty -prune -o -name '*.h' -exec clang-format -style=file --verbose -i "{}" \; -print
        find src/ -path src/thirdparty -prune -o -name '*.cpp' -exec clang-format -style=file --verbose -i "{}" \; -print
        find src/ -name "*.cpp" -print0 | xargs -0 -r clang-tidy -p temp/build
        find src/ -name "*.h" -print0 | xargs -0 -r clang-tidy -p temp/build
        find src/ -name "*.hpp" -print0 | xargs -0 -r clang-tidy -p temp/build
        ;;
    doxygen) # generate doxygen documentation
        rm -rf temp/documentations/doxygen
        mkdir -p temp/documentations/doxygen
        doxygen .doxyfile
        mv -T temp/documentations/html temp/documentations/doxygen
        ;;
    build) # build
        rm -rf temp/build
        cmake -S . -B temp/build -DQT_LIB_PATH=$QT_LIB_PATH -DCMAKE_BUILD_TYPE=Debug
        make --directory=temp/build -j$(nproc) all install
        ;;
    test) # test
        # valgrind --tool=memcheck --leak-check=yes ./temp/build/bin/MainBenchmarks --benchmark_filter=LatticeBoltzmannMethodD2Q9_InitiationBenchmark
        rm -rf temp/documentations/coverages
        mkdir -p temp/documentations/coverages
        temp/build/bin/MainTests
        temp/build/bin/MainBenchmarks --benchmark_time_unit=ms
        gcovr -r . -e '.*\.moc' -e '.*Tests.*' -e '.*Test.*' --exclude-unreachable-branches --exclude-throw-branches --html --html-details -o temp/documentations/coverages/report_coverage.html
        ;;
    run) # run
        temp/build/bin/FastFluidDynamicSimulationForGames
        ;;
    *)
        echo "Invalid Commandule specified: $Command" 
        exit 1
esac