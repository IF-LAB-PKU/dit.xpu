#include "htp_dense_mask.h"
#include "ggml-cpu.h"
#ifdef SD_TEST_HTP
#include "ggml-htp.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static ggml_context* context() {
    return ggml_init({2 * 1024 * 1024, nullptr, true});
}

// Evaluate the pre-cache SCALE -> REPEAT -> CONCAT -> F16 chain.
static std::vector<ggml_fp16_t> legacy(ggml_backend_t cpu, int64_t queries, int64_t keys) {
    auto* ctx = context();
    auto* one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto* mask = ggml_repeat_4d(ctx, ggml_scale(ctx, one, 0.0f), keys, queries, 1, 1);
    const int64_t stride = GGML_PAD(keys, 64);
    if (stride > keys) {
        auto* pad = ggml_repeat_4d(ctx, ggml_scale(ctx, one, -INFINITY), stride - keys, queries, 1, 1);
        mask = ggml_concat(ctx, mask, pad, 0);
    }
    mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mask);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buffer);
    const float value = 1.0f;
    ggml_backend_tensor_set(one, &value, 0, sizeof(value));
    assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    std::vector<ggml_fp16_t> result(ggml_nelements(mask));
    ggml_backend_tensor_get(mask, result.data(), 0, ggml_nbytes(mask));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void equivalence(ggml_backend_t cpu, ggml_backend_buffer_type_t buft) {
    HtpDenseMaskCache cache;
    for (int64_t image_tokens : {256, 1024, 4096}) {
        cache.begin_graph();
        auto* ctx = context();
        for (int64_t keys : {image_tokens, image_tokens + 154}) {
            auto expected = legacy(cpu, keys, keys);
            auto* mask = cache.get(ctx, buft, keys, keys, GGML_PAD(keys, 64));
            assert(mask && mask->type == GGML_TYPE_F16);
            assert(std::memcmp(mask->data, expected.data(), ggml_nbytes(mask)) == 0);
            auto* again = cache.get(ctx, buft, keys, keys, GGML_PAD(keys, 64));
            assert(again->data == mask->data);
        }
        cache.end_graph();
        assert(cache.size() == 2);
        auto* last = cache.get(ctx, buft, image_tokens, image_tokens, image_tokens);
        void* backing = last->data;
        ggml_free(ctx);
        ctx = context();
        cache.begin_graph();
        last = cache.get(ctx, buft, image_tokens, image_tokens, image_tokens);
        assert(last->data == backing);
        // A graph allocator must preserve the independently owned input.
        auto* out = ggml_cast(ctx, last, GGML_TYPE_F32);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        assert(ggml_gallocr_alloc_graph(alloc, graph));
        assert(last->data == backing);
        assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
        ggml_gallocr_free(alloc);
        cache.end_graph();
        assert(cache.size() == 1);
        ggml_free(ctx);
    }
    auto* ctx = context();
    assert(!cache.get(ctx, buft, 0, 64, 64));
    assert(!cache.get(ctx, buft, 64, 65, 64));
    assert(!cache.get(ctx, buft, 64, 64, 65));
    assert(!cache.get(ctx, buft, INT64_MAX, 64, 64));
    cache.begin_graph();
    auto* a = cache.get(ctx, buft, 32, 63, 64);
    auto* b = cache.get(ctx, buft, 64, 63, 64);
    assert(a && b && a->data != b->data);
    cache.end_graph();
    assert(cache.size() == 2);
    ggml_free(ctx);
    cache.clear();
    assert(cache.size() == 0);
    std::puts("PASS: legacy bit equivalence at 256/512/1024, reuse, allocator, shape pruning, validation");
}

static void attention(ggml_backend_t cpu, ggml_backend_t htp, int64_t keys) {
    HtpDenseMaskCache cache;
    auto* ctx = context();
    const int64_t queries = keys;
    const int64_t stride = GGML_PAD(keys, 64);
    auto buft = ggml_backend_get_default_buffer_type(htp);
    auto* mask = cache.get(ctx, buft, queries, keys, stride);
    auto* old_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, stride, queries);
    auto* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, queries, 2);
    auto* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, keys, 2);
    auto* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, keys, 2);
    auto* old_out = ggml_flash_attn_ext(ctx, q, k, v, old_mask, 0.125f, 0, 0);
    auto* new_out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0, 0);
    ggml_flash_attn_ext_set_prec(old_out, GGML_PREC_F32);
    ggml_flash_attn_ext_set_prec(new_out, GGML_PREC_F32);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, old_out);
    ggml_build_forward_expand(graph, new_out);
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    assert(buffer && mask->buffer != buffer);
    auto expected = legacy(cpu, queries, keys);
    ggml_backend_tensor_set(old_mask, expected.data(), 0, ggml_nbytes(old_mask));
    std::vector<float> q_data(ggml_nelements(q));
    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = std::sin(float(i) * 0.03f);
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    for (auto* tensor : {k, v}) {
        std::vector<ggml_fp16_t> data(ggml_nelements(tensor));
        for (size_t i = 0; i < data.size(); ++i) data[i] = ggml_fp32_to_fp16(std::cos(float(i) * 0.02f));
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }
    assert(ggml_backend_graph_compute(htp, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> a(ggml_nelements(old_out)), b(a.size());
    ggml_backend_tensor_get(old_out, a.data(), 0, ggml_nbytes(old_out));
    ggml_backend_tensor_get(new_out, b.data(), 0, ggml_nbytes(new_out));
    for (size_t i = 0; i < a.size(); ++i) assert(std::isfinite(a[i]) && a[i] == b[i]);
    std::printf("PASS: HTP attention keys=%lld old/cached output bit-identical\n", (long long)keys);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char**) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef GGML_BACKEND_DL
    ggml_backend_load_all();
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
#else
    auto cpu = ggml_backend_cpu_init();
#endif
    assert(cpu);
#ifndef GGML_BACKEND_DL
    ggml_backend_cpu_set_n_threads(cpu, 4);
#endif
    ggml_backend_t htp = nullptr;
#ifdef SD_TEST_HTP
    if (argc > 1) htp = ggml_backend_htp_init();
#else
    if (argc > 1) htp = ggml_backend_init_by_name("HTP", nullptr);
#endif
    if (argc > 1) assert(htp);
    equivalence(cpu, ggml_backend_get_default_buffer_type(htp ? htp : cpu));
    if (htp) {
        attention(cpu, htp, 256);
        attention(cpu, htp, 410);
        ggml_backend_free(htp);
    }
    ggml_backend_free(cpu);
}
