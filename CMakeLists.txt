cmake_minimum_required(VERSION 3.4...3.18)
project(rgaspy)

add_subdirectory(extern/pybind11)
include_directories(extern/eigen)
pybind11_add_module(rgaspy src/pybind_functions.cpp)

# RGASPY_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(rgaspy
                           PRIVATE VERSION_INFO=${RGASPY_VERSION_INFO})