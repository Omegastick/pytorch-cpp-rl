target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/a2c.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/a2c.cpp
    )
endif (CPPRL_BUILD_TESTS)