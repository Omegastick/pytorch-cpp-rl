add_executable(gym_client gym_client.cpp communicator.cpp)

find_package(nlohmann_json)

target_include_directories(gym_client 
    PRIVATE
    .
    ../include
    ../lib/msgpack-c/include
    ../lib/spdlog/include
)

target_link_libraries(gym_client PRIVATE zmq cpprl nlohmann_json::nlohmann_json)