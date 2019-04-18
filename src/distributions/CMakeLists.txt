target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/categorical.cpp
    ${CMAKE_CURRENT_LIST_DIR}/normal.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/categorical.cpp
        ${CMAKE_CURRENT_LIST_DIR}/normal.cpp
    )
endif (CPPRL_BUILD_TESTS)