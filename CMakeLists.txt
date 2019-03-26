cmake_minimum_required(VERSION 3.10.0)

project(CppRl
    LANGUAGES CXX
    VERSION 0.1.0
    DESCRIPTION "Reinforcement learning in C++ using PyTorch"
)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_library(cpprl STATIC "")
option(CPPRL_BUILD_TESTS "Whether or not to build the CppRl tests" ON)
if (CPPRL_BUILD_TESTS)
    add_executable(cpprl_tests "")
endif(CPPRL_BUILD_TESTS)

if(MSVC)
    target_compile_options(cpprl PRIVATE /W4 /WX)
else(MSVC)
    target_compile_options(cpprl PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)  

set(CPPRL_INCLUDE_DIRS
    include
    src
    ${TORCH_INCLUDE_DIRS}
)

target_include_directories(cpprl PRIVATE ${CPPRL_INCLUDE_DIRS})
target_include_directories(cpprl_tests PRIVATE ${CPPRL_INCLUDE_DIRS})

target_link_libraries(cpprl torch ${TORCH_LIBRARIES})
target_link_libraries(cpprl_tests torch ${TORCH_LIBRARIES})

if (TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
endif()

add_subdirectory(src)