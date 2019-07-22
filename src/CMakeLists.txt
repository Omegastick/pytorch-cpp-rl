target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/observation_normalizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/running_mean_std.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/storage.cpp
        ${CMAKE_CURRENT_LIST_DIR}/observation_normalizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/running_mean_std.cpp
    )
endif (CPPRL_BUILD_TESTS)

add_subdirectory(algorithms)
add_subdirectory(distributions)
add_subdirectory(generators)
add_subdirectory(model)
add_subdirectory(third_party)