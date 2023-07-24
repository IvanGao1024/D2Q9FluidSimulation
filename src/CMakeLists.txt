###QT
#load all QT6 lib
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
find_package(Qt6 REQUIRED COMPONENTS WebSockets)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS Multimedia)
# find_package(Qt6 COMPONENTS Network REQUIRED)
# find_package(Qt6 COMPONENTS Concurrent REQUIRED)
# find_package(Qt6 COMPONENTS Sql REQUIRED)

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

###link QT lib
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Qt6::Core)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Qt6::Gui)
target_link_libraries(${PROJECT_EXECUTABLE_NAME} PRIVATE Qt6::Widgets)