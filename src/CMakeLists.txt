target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/model.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/model.cpp
    )
endif (CPPRL_BUILD_TESTS)


add_subdirectory(third_party)