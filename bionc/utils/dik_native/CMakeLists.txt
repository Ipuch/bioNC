cmake_minimum_required(VERSION 3.18)
project(dik_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

nanobind_add_module(dik_native STABLE_ABI dik_native.cpp)
target_link_libraries(dik_native PRIVATE Eigen3::Eigen ${CMAKE_DL_LIBS})
target_compile_options(dik_native PRIVATE -O3 -march=native -funroll-loops -DNDEBUG)
