set(TEST_TARGET test-model-loader-subset)
add_executable(${TEST_TARGET} test-model-loader-subset.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

set(TEST_TARGET test-convert-f16)
add_executable(${TEST_TARGET} test-convert-f16.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-denoise)
add_executable(${TEST_TARGET}
               ../examples/common/common.cpp
               ../examples/common/log.cpp
               test-denoise.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE
                           ${CMAKE_CURRENT_LIST_DIR}/..
                           ${CMAKE_CURRENT_LIST_DIR}/../examples
                           ${CMAKE_CURRENT_LIST_DIR}/../src)

set(TEST_TARGET test-htp-route-policy)
add_executable(${TEST_TARGET} test-htp-route-policy.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

set(TEST_TARGET test-backend-htp-residual-cache)
add_executable(${TEST_TARGET} ../ggml/tests/${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
if (APPLE)
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,-multiply_defined" "-Wl,suppress")
else()
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,--allow-multiple-definition")
endif()
add_test(NAME ${TEST_TARGET}-fp16 COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET}-fp16 PROPERTY
             ENVIRONMENT "HTP_RESIDUAL_CACHE_TEST=fp16;LLVM_PROFILE_FILE=${TEST_TARGET}-fp16.profraw")
add_test(NAME ${TEST_TARGET}-lifecycle COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET}-lifecycle PROPERTY
             ENVIRONMENT "HTP_RESIDUAL_CACHE_TEST=lifecycle;LLVM_PROFILE_FILE=${TEST_TARGET}-lifecycle.profraw")
set_property(TEST ${TEST_TARGET}-lifecycle PROPERTY SKIP_RETURN_CODE 77)
add_test(NAME ${TEST_TARGET}-graph-order COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET}-graph-order PROPERTY
             ENVIRONMENT "HTP_RESIDUAL_CACHE_TEST=graph-order;LLVM_PROFILE_FILE=${TEST_TARGET}-graph-order.profraw")
add_test(NAME ${TEST_TARGET}-residual-only COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET}-residual-only PROPERTY
             ENVIRONMENT "HTP_RESIDUAL_CACHE_TEST=residual-only;LLVM_PROFILE_FILE=${TEST_TARGET}-residual-only.profraw")
set_property(TEST ${TEST_TARGET}-residual-only PROPERTY SKIP_RETURN_CODE 77)

set(TEST_TARGET test-backend-htp-attention-correctness)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-htp-attention-correctness.cpp
               ../ggml/tests/test-backend-htp-attention-common.cpp)
if (NOT GGML_HTP)
    target_sources(${TEST_TARGET} PRIVATE test-htp-host-stubs.cpp)
endif()
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

set(TEST_TARGET test-backend-htp-attention-kernel-performance)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-htp-attention-kernel-performance.cpp
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

set(TEST_TARGET test-backend-opencl-attention-kernel-performance)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-opencl-attention-kernel-performance.cpp
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

set(TEST_TARGET test-backend-opencl-attention-correctness)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-opencl-attention-correctness.cpp
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
