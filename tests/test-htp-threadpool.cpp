// Device regression/benchmark for repeated graph cuts and pool replacement.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-htp.h"
#include "htp-hybrid.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool abort_graph(void *) { return true; }

int main() {
    auto * htp = ggml_backend_init_by_name("HTP", nullptr);
    auto * cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(htp && cpu);
    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_cpu_set_n_threads(cpu, 4);

    for (int rows : {1, 77, 257}) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, rows);
        auto * y = ggml_scale(ctx, ggml_add(ctx, ggml_sqr(ctx, x), x), .5f);
        auto * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, y);
        // These fallback-only nodes deliberately exercise HTP's CPU workers.
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        GGML_ASSERT(buffer);
        std::vector<float> input(ggml_nelements(x)), expected(input.size()), actual(input.size());
        for (int threads : {4, 4, 1, 2, 4, 1}) {
            ggml_backend_htp_set_n_threads(htp, threads);
            for (int repeat = 0; repeat < 6; ++repeat) {
                for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15 + repeat) / 16;
                ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
                GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_get(y, expected.data(), 0, ggml_nbytes(y));
                for (const char * mode : {"0", "1"}) {
                    setenv("SD_HTP_PERSISTENT_THREADPOOL", mode, 1);
                    GGML_ASSERT(ggml_backend_graph_compute(htp, graph) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
                    GGML_ASSERT(actual == expected);
                }
            }
        }

        // A cancelled plan must not poison a persistent pool's next graph.
        auto params = ggml_threadpool_params_default(4);
        params.paused = true;
        auto * pool = ggml_htp_threadpool_new(&params);
        auto plan = ggml_htp_graph_plan(graph, 4, pool);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        for (int repeat = 0; repeat < 32; ++repeat) {
            plan.n_threads = repeat % 2 ? 1 : 4;
            plan.abort_callback = abort_graph;
            GGML_ASSERT(ggml_graph_compute_htp_hybrid(graph, &plan) == GGML_STATUS_ABORTED);
            plan.abort_callback = nullptr;
            GGML_ASSERT(ggml_graph_compute_htp_hybrid(graph, &plan) == GGML_STATUS_SUCCESS);
        }
        ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
        GGML_ASSERT(actual == expected);
        ggml_htp_threadpool_free(pool);
        std::printf("lifecycle rows=%d exact=1 cancel_recovery=1\n", rows);

        if (rows == 77) {
            ggml_backend_htp_set_n_threads(htp, 4);
            // Alternate order across rounds; time only graph compute, not I/O.
            for (int round = 0; round < 4; ++round) {
                for (int order = 0; order < 2; ++order) {
                    int mode = (round + order) % 2;
                    setenv("SD_HTP_PERSISTENT_THREADPOOL", mode ? "1" : "0", 1);
                    GGML_ASSERT(ggml_backend_graph_compute(htp, graph) == GGML_STATUS_SUCCESS);
                    auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < 200; ++i) {
                        GGML_ASSERT(ggml_backend_graph_compute(htp, graph) == GGML_STATUS_SUCCESS);
                    }
                    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                    std::printf("benchmark round=%d persistent=%d graphs=200 us_per_graph=%.3f\n", round, mode, us / 200);
                    std::fflush(stdout);
                }
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(htp);
    ggml_backend_free(cpu);
    std::puts("HTP threadpool lifecycle PASSED");
}
