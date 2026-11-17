set(TEST_TARGET test-model-loader-subset)
add_executable(${TEST_TARGET} test-model-loader-subset.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

if(UNIX OR ANDROID)
    set(TEST_TARGET test-htp-cli-defaults)
    add_executable(${TEST_TARGET} test-htp-cli-defaults.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion)
    target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

set(TEST_TARGET test-text-offline-pack)
add_executable(${TEST_TARGET} test-text-offline-pack.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-graph-cut-views)
add_executable(${TEST_TARGET} test-graph-cut-views.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-z-image-graph-cuts)
add_executable(${TEST_TARGET} test-z-image-graph-cuts.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

if(GGML_HTP)
    set(TEST_TARGET test-htp-packed-fallback)
    add_executable(${TEST_TARGET} test-htp-packed-fallback.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion)
    target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
endif()

# Explicit real-model/device benchmark, not part of ordinary CTest runs.
set(TEST_TARGET test-z-image-forward)
add_executable(${TEST_TARGET} test-z-image-forward.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)

set(TEST_TARGET test-text-encoder)
add_executable(${TEST_TARGET} test-text-encoder.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)

set(TEST_TARGET test-htp-message-layout)
add_executable(${TEST_TARGET} ../ggml/src/ggml-htp/message_layout_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-packed-staging)
add_executable(${TEST_TARGET} test-htp-packed-staging.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

set(TEST_TARGET test-htp-dense-mask-cache)
add_executable(${TEST_TARGET} test-htp-dense-mask-cache.cpp)
if(GGML_HTP AND NOT GGML_BACKEND_DL)
    target_compile_definitions(${TEST_TARGET} PRIVATE SD_TEST_HTP)
endif()
target_link_libraries(${TEST_TARGET} PRIVATE ggml Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-backend-htp-packed-qkv-staging)
add_executable(${TEST_TARGET} ../ggml/tests/${TEST_TARGET}.cpp)
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

set(TEST_TARGET test-htp-graph-cut-session-budget)
add_executable(${TEST_TARGET} test-htp-graph-cut-session-budget.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-tensor-session-view)
add_executable(${TEST_TARGET} ../ggml/src/ggml-htp/tensor_session_view_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-session-placement)
add_executable(${TEST_TARGET}
               ../ggml/src/ggml-htp/session_placement_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-session-placement-plan)
add_executable(${TEST_TARGET}
               ../ggml/src/ggml-htp/session_placement_plan_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-session-param-placement)
add_executable(${TEST_TARGET}
               ../ggml/src/ggml-htp/session_param_placement_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-session-dispatch)
add_executable(${TEST_TARGET}
               ../ggml/src/ggml-htp/session_dispatch_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-rpcmem-mapper-range)
add_executable(${TEST_TARGET}
               ../ggml/src/ggml-htp/rpcmem_mapper_range_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/include)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-htp-graph-cut-session-dispatch)
add_executable(${TEST_TARGET} test-htp-graph-cut-session-dispatch.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)


set(TEST_TARGET test-htp-rpcmem-session-route)
add_executable(${TEST_TARGET}
               ../ggml/src/ggml-htp/rpcmem_session_route_test.cc)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-backend-htp-opencl-branch-modes)
add_executable(${TEST_TARGET} ../ggml/tests/${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
if (APPLE)
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,-multiply_defined" "-Wl,suppress")
else()
    target_link_options(${TEST_TARGET} PRIVATE "-Wl,--allow-multiple-definition")
endif()
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY
             ENVIRONMENT "HTP_OPENCL_BRANCH_MODES=all;LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

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
add_test(NAME ${TEST_TARGET}-sparse-residual-preflight
         COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET}-sparse-residual-preflight PROPERTY
             ENVIRONMENT "HTP_RESIDUAL_CACHE_TEST=sparse-residual-preflight;LLVM_PROFILE_FILE=${TEST_TARGET}-sparse-residual-preflight.profraw")
set_property(TEST ${TEST_TARGET}-sparse-residual-preflight PROPERTY SKIP_RETURN_CODE 77)

if(GGML_HTP AND NOT TARGET test-backend-htp-matmul-f16)
    add_executable(test-backend-htp-linear-performance test-backend-htp-linear-performance.cpp)
    target_link_libraries(test-backend-htp-linear-performance PRIVATE ggml Threads::Threads)
    add_executable(test-backend-htp-matmul-f16
                   ../ggml/tests/test-backend-htp-matmul-f16.cpp)
    target_link_libraries(test-backend-htp-matmul-f16 PRIVATE ggml Threads::Threads)
    add_test(NAME test-backend-htp-matmul-f16 COMMAND test-backend-htp-matmul-f16)
endif()

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

set(TEST_TARGET test-backend-htp-mask-to-attention-performance)
add_executable(${TEST_TARGET}
               ../ggml/tests/${TEST_TARGET}.cpp
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

set(TEST_TARGET test-backend-cpu-attention-kernel-performance)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-cpu-attention-kernel-performance.cpp
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

set(TEST_TARGET test-backend-cpu-sparse-attention-kernel-performance)
add_executable(${TEST_TARGET}
               ../ggml/tests/test-backend-cpu-sparse-attention-kernel-performance.cpp
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

add_test(NAME test-htp-host-release-build-contract
         COMMAND sh
                 ${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-htp/test-release-build-contract.sh
                 ${CMAKE_BINARY_DIR}/CMakeCache.txt
                 ${CMAKE_BINARY_DIR}/compile_commands.json)

set(TEST_TARGET test-model-families)
add_executable(${TEST_TARGET} test-model-families.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

# Explicit device experiment; requires a VAE model or a functioning HTP runtime.
set(TEST_TARGET test-vae-decode)
add_executable(${TEST_TARGET} test-vae-decode.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../src)
target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)

set(TEST_TARGET test-vae-offline-pack)
add_executable(${TEST_TARGET} test-vae-offline-pack.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE stable-diffusion Threads::Threads)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

if(GGML_HTP)
    # Explicit device correctness test; needs a matched host/DSP runtime.
    add_executable(test-htp-threadpool test-htp-threadpool.cpp)
    target_link_libraries(test-htp-threadpool PRIVATE ggml ggml-htp)
    target_include_directories(test-htp-threadpool PRIVATE ../ggml/src/ggml-htp)
    target_compile_options(test-htp-threadpool PRIVATE -UNDEBUG)
    add_executable(test-htp-native test-htp-native.cpp)
    target_link_libraries(test-htp-native PRIVATE ggml)
    target_compile_options(test-htp-native PRIVATE -UNDEBUG)
    add_executable(test-htp-dit test-htp-dit.cpp)
    target_link_libraries(test-htp-dit PRIVATE ggml)
    target_compile_options(test-htp-dit PRIVATE -UNDEBUG)
    add_executable(test-htp-f16-fusion test-htp-f16-fusion.cpp)
    target_link_libraries(test-htp-f16-fusion PRIVATE ggml)
    target_compile_options(test-htp-f16-fusion PRIVATE -UNDEBUG)
endif()
