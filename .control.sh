#!/bin/bash
# environment variable

# QT_LIB_PATH=/6.2.0/gcc_64/lib
# QT_LIB_PATH=/home/ivan/QT6.5.1/6.5.1/gcc_64/lib

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
    -command=doc)
    Command=doc
    shift # past argument=value
    ;;
    -command=buildClient)
    Command=buildClient
    shift # past argument=value
    ;;
    -command=buildServer)
    Command=buildServer
    shift # past argument=value
    ;;
    -command=deploy)
    Command=deploy
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
        find src/ -name "*.cpp" | xargs clang-tidy -p temp/build
        find src/ -name "*.h" | xargs clang-tidy -p temp/build
        find src/ -name "*.hpp" | xargs clang-tidy -p temp/build
        ;;
    doc) # build both document and code document
        # generate doxygen documentation
        mkdir -p temp/doc
        doxygen .doxyfile
        mv -T temp/doc/html temp/doc/doxygen
        ;;
    buildClient) # build client
        rm -rf temp/deploy/client
        cmake -S . -B temp/build -DQT_LIB_PATH=$QT_LIB_PATH -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEBUG=YES -DBUILD_TESTS=NO -DBUILD_SERVER=NO -DBUILD_CLIENT=YES
        make --directory=temp/build -j$(nproc) all install
        case $Platform in
        linux)
            echo -e "#!/bin/bash\nexport LD_LIBRARY_PATH=${QT_LIB_PATH}\n./CreepsWorldClient" > temp/deploy/client/startup.sh
            ;;
        windows)
            ;;
        mac)
            ;;
        *)
            echo "Invalid Platform specified: $Platform"
            exit 1
        esac
        ;;
    buildServer) # build server
        rm -rf temp/deploy/server
        cmake -S . -B temp/build -DQT_LIB_PATH=$QT_LIB_PATH -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEBUG=YES -DBUILD_TESTS=NO -DBUILD_SERVER=YES -DBUILD_CLIENT=NO
        make --directory=temp/build -j$(nproc) all install
        case $Platform in
        linux)
            echo -e "#!/bin/bash\nexport LD_LIBRARY_PATH=${QT_LIB_PATH}\n./CreepsWorldServer" > temp/deploy/server/startup.sh
            ;;
        windows)
            ;;
        mac)
            ;;
        *)
            echo "Invalid Platform specified: $Platform"
            exit 1
        esac
        ;;
    deploy) # deploy both server and client
        rm -rf temp/deploy/
        cmake -S . -B temp/build -DQT_LIB_PATH=$QT_LIB_PATH -DCMAKE_BUILD_TYPE=release -DENABLE_DEBUG=YES -DBUILD_TESTS=NO -DBUILD_SERVER=YES -DBUILD_CLIENT=YES
        make --directory=temp/build -j$(nproc) all install
        case $Platform in
        linux)
            echo -e "#!/bin/bash\nexport LD_LIBRARY_PATH=${QT_LIB_PATH}\n./CreepsWorldClient" > temp/deploy/client/startup.sh
            echo -e "#!/bin/bash\nexport LD_LIBRARY_PATH=${QT_LIB_PATH}\n./CreepsWorldServer" > temp/deploy/server/startup.sh
            ;;
        windows)
            ;;
        mac)
            ;;
        *)
            echo "Invalid Platform specified: $Platform"
            exit 1
        esac
        ;;
    # test) # build test and report coverage
        # TODO:
        # cmake -S . -B ./temp/build -DQT_LIB_PATH=$QT_LIB_PATH -DCMAKE_BUILD_TYPE=Debug -DDEBUG=YES -DBUILD_TESTS=YES -DBUILD_SERVER=NO -DBUILD_CLIENT=NO
        # make --directory=temp/build -j7
        # ctest --test-dir ./temp/build --output-on-failure --parallel 4
        #!/bin/sh

        # coverage
        # cd runtime
        # rm -rf coverages
        # mkdir -p coverages
        # gcovr -r ./.. -e '.*\.moc' -e '.*Tests.*' -e '.*Test.*' --exclude-unreachable-branches --exclude-throw-branches --xml -o ./coverages/coverages.xml
        # gcovr -r ./.. -e '.*\.moc' -e '.*Tests.*' -e '.*Test.*' --exclude-unreachable-branches --exclude-throw-branches --html --html-details -o ./coverages/coverages.html
        # ;;
        # GDB test code: gdb -ex "set env LD_LIBRARY_PATH /home/ivan/QT6.5.1/6.5.1/gcc_64/lib" -ex "run" ./CreepsWorldClient
    *)
        echo "Invalid Commandule specified: $Command" 
        exit 1
esac