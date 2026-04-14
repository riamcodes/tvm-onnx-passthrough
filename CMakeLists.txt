cmake_minimum_required(VERSION 3.18)
project(tvm_onnx_passthrough LANGUAGES CXX)

set(PYBIND11_FINDPYTHON OFF)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native cpp/module.cpp)
target_include_directories(_native PRIVATE /usr/local/es7/include/python3.9)

install(TARGETS _native DESTINATION tvm_onnx_passthrough)
