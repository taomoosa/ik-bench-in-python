cmake_minimum_required(VERSION 3.14)
project(quik_server_client CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED NO_MODULE)

set(QUIK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/quik" CACHE PATH "Path to quik source directory")
set(QUIK_INCLUDE_DIR ${QUIK_DIR}/include)
set(QUIK_GEOMETRY_SRC ${QUIK_DIR}/src/geometry.cpp)

foreach(DOF 6 7)
    add_executable(quik_server_${DOF}dof server.cpp ${QUIK_GEOMETRY_SRC})
    target_compile_definitions(quik_server_${DOF}dof PRIVATE ROBOT_DOF=${DOF})
    target_include_directories(quik_server_${DOF}dof PRIVATE ${QUIK_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(quik_server_${DOF}dof Eigen3::Eigen)
endforeach()
