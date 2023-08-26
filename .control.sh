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
        rm -rf temp/documentations/profile
        mkdir -p temp/documentations/coverages
        mkdir -p temp/documentations/profile

        # Run the tests and generate gmon.out
        # Check if MainTests exists and is executable
        if [ ! -x "temp/build/bin/MainTests" ]; then
            echo "MainTests not found or is not executable. Exiting."
            exit 1
        fi
        temp/build/bin/MainTests
        temp/build/bin/MainBenchmarks --benchmark_time_unit=ms

        # Generate the coverage report
        gcovr -r . -e '.*\.moc' -e '.*Tests.*' -e '.*Test.*' --exclude-unreachable-branches --exclude-throw-branches --html --html-details -o temp/documentations/coverages/report_coverage.html

        # Move gmon.out and gprof directory to the desired directory
        mv gmon.out temp/build/

        # Check if the virtual environment already exists, if not create it
        if [ ! -d "temp/python3/gprof_venv" ]; then
            python3 -m venv temp/python3/gprof_venv
        fi

        # Activate the virtual environment
        source temp/python3/gprof_venv/bin/activate  # On Windows, use `temp/build/gprof_venv\Scripts\activate`

        # Install gprof2dot
        pip install gprof2dot

        # Generate the gprof output and save it as text
        gprof temp/build/bin/MainTests temp/build/gmon.out > temp/documentations/profile/gprof_analysis.txt

        # Generate a dot graph, filtering out functions that take less than 1ms
        gprof temp/build/bin/MainTests temp/build/gmon.out | gprof2dot -n 1 -e 0 | dot -Tpng -o temp/documentations/profile/profile.png

        # Deactivate the virtual environment
        deactivate
        ;;
    run) # run
        temp/build/bin/FastFluidDynamicSimulationForGames
        ;;
    *)
        echo "Invalid Commandule specified: $Command" 
        exit 1
esac