target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cnnbase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlpbase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nnbase.cpp
    ${CMAKE_CURRENT_LIST_DIR}/policy.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cnnbase.cpp
        ${CMAKE_CURRENT_LIST_DIR}/mlpbase.cpp
        ${CMAKE_CURRENT_LIST_DIR}/model_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/nnbase.cpp
        ${CMAKE_CURRENT_LIST_DIR}/policy.cpp
    )
endif (CPPRL_BUILD_TESTS)