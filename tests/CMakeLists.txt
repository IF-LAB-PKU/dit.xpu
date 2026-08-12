set(TEST_TARGET test-model-loader-subset)
add_executable(${TEST_TARGET} test-model-loader-subset.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

set(TEST_TARGET test-backend-htp-attention-correctness)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-htp-attention-correctness.cpp
               ../ggml/tests/test-backend-htp-attention-common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
if (APPLE)
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,-multiply_defined" "-Wl,suppress")
else()
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,--allow-multiple-definition")
endif()
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../include)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

set(TEST_TARGET test-backend-htp-attention-performance)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-htp-attention-performance.cpp
               ../ggml/tests/test-backend-htp-attention-common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
if (APPLE)
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,-multiply_defined" "-Wl,suppress")
else()
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,--allow-multiple-definition")
endif()
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../include)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

add_test(NAME test-check-staged-weight-log
         COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/test-check-staged-weight-log.sh)
