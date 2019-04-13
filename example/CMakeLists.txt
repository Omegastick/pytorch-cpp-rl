add_executable(gym_client gym_client.cpp communicator.cpp)

set(LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CPPZMQ_DIR ${LIB_DIR}/cppzmq)
set(MSGPACK_DIR ${LIB_DIR}/msgpack-c)
set(ZMQ_DIR ${LIB_DIR}/libzmq)

# ZMQ
set(ZMQ_BUILD_TESTS OFF)
add_subdirectory(${ZMQ_DIR})

target_include_directories(gym_client 
    PRIVATE
    .
    ../include
    ../lib/spdlog/include
    ${CPPZMQ_DIR}
    ${MSGPACK_DIR}/include
    ${ZMQ_DIR}/include
)

target_link_libraries(gym_client PRIVATE libzmq-static cpprl)