cmake_minimum_required(VERSION 3.10.0)

project(CppRl
    LANGUAGES CXX
    VERSION 0.1.0
    DESCRIPTION "Reinforcement learning in C++ using PyTorch"
)

find_package(Torch REQUIRED)

add_library(CppRl SHARED)

if(MSVC)
    target_compile_options(CppRl PRIVATE /W4 /WX)
else(MSVC)
    target_compile_options(CppRl PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)  

target_include_directories(CppRl
    PRIVATE
    include
    src
)

add_subdirectory(src)