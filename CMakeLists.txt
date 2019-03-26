cmake_minimum_required(VERSION 3.10.0)

project(CppRl
    LANGUAGES CXX
    VERSION 0.1.0
    DESCRIPTION "Reinforcement learning in C++ using PyTorch"
)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_library(cpprl STATIC "")

if(MSVC)
    target_compile_options(cpprl PRIVATE /W4 /WX)
else(MSVC)
    target_compile_options(cpprl PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)  

target_include_directories(cpprl
    PRIVATE
    include
    src
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cpprl torch ${TORCH_LIBRARIES})

message(STATUS "${TORCH_CXX_FLAGS}")
if (TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
endif()

add_subdirectory(src)