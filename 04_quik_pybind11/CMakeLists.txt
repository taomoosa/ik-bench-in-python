cmake_minimum_required(VERSION 3.14)
project(quik_pybind11 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)

set(QUIK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/quik" CACHE PATH "Path to quik source directory")
set(QUIK_INCLUDE_DIR ${QUIK_DIR}/include)

pybind11_add_module(quik_binding
    quik_binding.cpp
    ${QUIK_DIR}/src/geometry.cpp
)

target_include_directories(quik_binding PRIVATE
    ${QUIK_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(quik_binding PRIVATE Eigen3::Eigen)
