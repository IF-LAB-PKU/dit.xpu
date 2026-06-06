// 验证 diffusion HTP 路由资格判定：只有 weight=F16, act=F32, dst=F32 的 MUL_MAT 会被视为 HTP row-major F16 目标算子。

#include "../../src/ggml_extend.hpp"

#include <ggml-backend.h>
#include <ggml-htp.h>
#include <ggml.h>

#include <cstdio>
#include <numeric>
#include <vector>

namespace {

struct RoutingProbeRunner : GGMLRunner {
    explicit RoutingProbeRunner(ggml_backend_t backend)
        : GGMLRunner(backend, backend) {
        // 这里直接把 CPU backend 伪装成 htp_backend，只验证 sd.cpp 侧的资格判定和图规划逻辑，不依赖真机 HTP runtime。
        set_htp_backend(backend);
    }

    std::string get_desc() override {
        return "htp-routing-probe";
    }

    ggml_context * init_compute_ctx() {
        reset_compute_ctx();
        return compute_ctx;
    }

    bool eligible_f16_mul_mat(ggml_tensor * node) const {
        return is_htp_row_major_f16_mul_mat(node);
    }

    bool htp_target(ggml_tensor * node) const {
        return is_htp_target_op(node);
    }

    std::vector<ggml_tensor *> planned_targets(ggml_cgraph * gf) const {
        return collect_planned_htp_target_ops(gf);
    }
};

struct PlacementProbeRunner : GGMLRunner {
    ggml_tensor * eligible_w0   = nullptr;
    ggml_tensor * ineligible_w1 = nullptr;
    ggml_tensor * eligible_w2   = nullptr;
    ggml_tensor * unmarked_w3   = nullptr;
    ggml_tensor * view_root     = nullptr;
    ggml_tensor * eligible_view = nullptr;
    std::vector<float> act0_data;
    std::vector<float> act1_data;
    std::vector<float> act2_data;

    PlacementProbeRunner(ggml_backend_t runtime_backend, ggml_backend_t params_backend, ggml_backend_t htp)
        : GGMLRunner(runtime_backend, params_backend) {
        // 这里的参数张量明确建在 params_ctx 里，模拟 diffusion 权重，后续检查它们是否会被一次性迁移到 RPCMEM。
        eligible_w0   = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F16, 64, 96);
        ineligible_w1 = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F32, 64, 96);
        eligible_w2   = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F16, 64, 96);
        unmarked_w3   = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F16, 64, 96);
        view_root     = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F16, 64, 384);
        GGML_ASSERT(eligible_w0 != nullptr && ineligible_w1 != nullptr && eligible_w2 != nullptr && unmarked_w3 != nullptr && view_root != nullptr);

        ggml_set_name(eligible_w0, "placement_eligible_w0");
        ggml_set_name(ineligible_w1, "placement_ineligible_w1");
        ggml_set_name(eligible_w2, "placement_eligible_w2");
        ggml_set_name(unmarked_w3, "placement_unmarked_w3");
        ggml_set_name(view_root, "placement_large_view_root");
        // 参照 llama.cpp-npu 的 weight-op 映射，只有明确会作为 MUL_MAT weight 使用的参数才允许预放置到 HTP。
        eligible_w0->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT;
        eligible_w2->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT;
        view_root->flags |= SD_GGML_TENSOR_FLAG_HTP_MUL_MAT_WEIGHT;
        set_htp_backend(htp);
    }

    std::string get_desc() override {
        return "htp-placement-probe";
    }

    ggml_cgraph * build_graph() {
        reset_compute_ctx();

        ggml_tensor * act0 = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * act1 = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * act2 = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
        GGML_ASSERT(act0 != nullptr && act1 != nullptr && act2 != nullptr);

        ggml_tensor * mm0  = ggml_mul_mat(compute_ctx, eligible_w0, act0);
        ggml_tensor * mm1  = ggml_mul_mat(compute_ctx, ineligible_w1, act1);
        ggml_tensor * mm2  = ggml_mul_mat(compute_ctx, eligible_w2, act2);
        ggml_tensor * sum0 = ggml_add(compute_ctx, mm0, mm1);
        ggml_tensor * sum  = ggml_add(compute_ctx, sum0, mm2);
        GGML_ASSERT(mm0 != nullptr && mm1 != nullptr && mm2 != nullptr && sum0 != nullptr && sum != nullptr);

        ggml_cgraph * gf = ggml_new_graph(compute_ctx);
        GGML_ASSERT(gf != nullptr);
        ggml_build_forward_expand(gf, sum);
        return gf;
    }

    std::vector<ggml_tensor *> directly_placed_htp_weights(ggml_cgraph * gf) const {
        (void)gf;
        std::vector<ggml_tensor *> tensors;
        if (htp_param_is_placed(eligible_w0)) {
            tensors.push_back(eligible_w0);
        }
        if (htp_param_is_placed(eligible_w2)) {
            tensors.push_back(eligible_w2);
        }
        if (htp_param_is_placed(view_root)) {
            tensors.push_back(view_root);
        }
        return tensors;
    }

    std::vector<ggml_tensor *> planned_targets(ggml_cgraph * gf) const {
        return collect_planned_htp_target_ops(gf);
    }

    void planned_buffer_id_sums(ggml_cgraph * gf, int * node_sum, int * leaf_sum) const {
        GGML_ASSERT(node_sum != nullptr);
        GGML_ASSERT(leaf_sum != nullptr);
        std::vector<int> node_ids;
        std::vector<int> leaf_ids;
        plan_compute_buffer_ids(gf, &node_ids, &leaf_ids);
        *node_sum = std::accumulate(node_ids.begin(), node_ids.end(), 0);
        *leaf_sum = std::accumulate(leaf_ids.begin(), leaf_ids.end(), 0);
    }

    ggml_cgraph * build_view_weight_graph() {
        reset_compute_ctx();

        GGML_ASSERT(view_root != nullptr);
        // 用一个明显大于实际 matmul 权重的 root 模拟模型里被切片引用的大参数，避免 HTP 持久化把整块 root 搬进 RPCMEM。
        eligible_view = ggml_view_2d(compute_ctx, view_root, 64, 96, view_root->nb[1], 128 * view_root->nb[1]);
        ggml_tensor * act = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
        GGML_ASSERT(eligible_view != nullptr && act != nullptr);
        ggml_set_name(eligible_view, "placement_eligible_view_weight");
        ggml_set_name(act, "placement_view_act");

        ggml_tensor * mm = ggml_mul_mat(compute_ctx, eligible_view, act);
        GGML_ASSERT(mm != nullptr);
        ggml_set_name(mm, "placement_view_mm");

        ggml_cgraph * gf = ggml_new_graph(compute_ctx);
        GGML_ASSERT(gf != nullptr);
        ggml_build_forward_expand(gf, mm);
        return gf;
    }

    ggml_backend_buffer_t get_default_params_buffer() const {
        return params_buffer;
    }

    ggml_backend_buffer_t get_direct_htp_params_buffer() const {
        return htp_param_placement_buffer_;
    }

    size_t runtime_offload_pair_count() const {
        return runtime_offload_pairs_.size();
    }

    size_t direct_htp_param_count() const {
        return htp_placed_param_tensors_.size();
    }

    bool direct_htp_contains(ggml_tensor * tensor) const {
        return htp_param_is_placed(tensor);
    }

    bool has_partial_runtime_params_buffer() const {
        return partial_runtime_params_buffer != nullptr;
    }

    std::optional<sd::Tensor<float>> run_fake_htp_compute() {
        // 这里用 CPU backend 伪装成 htp_backend，专门验证 HTP 参数不会再额外进入 partial runtime staging。
        act0_data.resize(64 * 4);
        act1_data.resize(64 * 4);
        act2_data.resize(64 * 4);
        for (size_t i = 0; i < act0_data.size(); ++i) {
            act0_data[i] = 0.01f * static_cast<float>((i % 13) + 1);
            act1_data[i] = 0.02f * static_cast<float>((i % 11) + 1);
            act2_data[i] = 0.03f * static_cast<float>((i % 7) + 1);
        }

        auto get_graph = [&]() -> ggml_cgraph * {
            ggml_cgraph * gf = ggml_new_graph(compute_ctx);
            GGML_ASSERT(gf != nullptr);

            ggml_tensor * act0 = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
            ggml_tensor * act1 = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
            ggml_tensor * act2 = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 64, 4);
            GGML_ASSERT(act0 != nullptr && act1 != nullptr && act2 != nullptr);

            ggml_set_name(act0, "fake_htp_act0");
            ggml_set_name(act1, "fake_htp_act1");
            ggml_set_name(act2, "fake_htp_act2");
            set_backend_tensor_data(act0, act0_data.data());
            set_backend_tensor_data(act1, act1_data.data());
            set_backend_tensor_data(act2, act2_data.data());

            ggml_tensor * mm0  = ggml_mul_mat(compute_ctx, eligible_w0, act0);
            ggml_tensor * mm1  = ggml_mul_mat(compute_ctx, ineligible_w1, act1);
            ggml_tensor * mm2  = ggml_mul_mat(compute_ctx, eligible_w2, act2);
            ggml_tensor * sum0 = ggml_add(compute_ctx, mm0, mm1);
            ggml_tensor * sum  = ggml_add(compute_ctx, sum0, mm2);
            GGML_ASSERT(mm0 != nullptr && mm1 != nullptr && mm2 != nullptr && sum0 != nullptr && sum != nullptr);

            ggml_build_forward_expand(gf, sum);
            return gf;
        };

        const char * old = std::getenv("SD_HTP_COMPUTE");
        std::string old_value = old != nullptr ? old : "";
        const bool had_old = old != nullptr;
        setenv("SD_HTP_COMPUTE", "1", 1);
        auto result = GGMLRunner::compute<float>(get_graph, 1, true);
        if (had_old) {
            setenv("SD_HTP_COMPUTE", old_value.c_str(), 1);
        } else {
            unsetenv("SD_HTP_COMPUTE");
        }
        return result;
    }
};

int find_node_index(ggml_cgraph * gf, ggml_tensor * tensor) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i) == tensor) {
            return i;
        }
    }
    return -1;
}

bool contains_tensor(const std::vector<ggml_tensor *> & tensors, ggml_tensor * needle) {
    for (ggml_tensor * tensor : tensors) {
        if (tensor == needle) {
            return true;
        }
    }
    return false;
}

bool tensor_is_rpcmem(ggml_tensor * tensor) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    return ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(tensor->buffer));
}

void fill_tensor_f16(ggml_tensor * tensor, float seed) {
    const int64_t count = ggml_nelements(tensor);
    std::vector<ggml_fp16_t> data(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        data[static_cast<size_t>(i)] = ggml_fp32_to_fp16(seed + 0.001f * static_cast<float>(i % 17));
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
}

void fill_tensor_f32(ggml_tensor * tensor, float seed) {
    const int64_t count = ggml_nelements(tensor);
    std::vector<float> data(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        data[static_cast<size_t>(i)] = seed + 0.001f * static_cast<float>(i % 19);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_t cpu_params = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu != nullptr);
    GGML_ASSERT(cpu_params != nullptr);

    RoutingProbeRunner runner(cpu);
    ggml_context * ctx = runner.init_compute_ctx();
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * act0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * w0   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 96);
    ggml_tensor * mm0  = ggml_mul_mat(ctx, w0, act0);

    ggml_tensor * act1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * w1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 96);
    ggml_tensor * mm1  = ggml_mul_mat(ctx, w1, act1);

    ggml_tensor * act2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * w2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 96);
    ggml_tensor * mm2  = ggml_mul_mat(ctx, w2, act2);

    ggml_tensor * act3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * w3   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 96);
    ggml_tensor * mm3  = ggml_mul_mat(ctx, w3, act3);
    ggml_tensor * rms  = ggml_rms_norm(ctx, act3, 1e-5f);
    ggml_tensor * act4 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 2);
    ggml_tensor * w4   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 96);
    ggml_tensor * mm4  = ggml_mul_mat(ctx, w4, act4);

    ggml_tensor * sum0 = ggml_add(ctx, mm0, mm2);
    ggml_tensor * sum  = ggml_add(ctx, sum0, mm3);

    GGML_ASSERT(act0 != nullptr && w0 != nullptr && mm0 != nullptr);
    GGML_ASSERT(act1 != nullptr && w1 != nullptr && mm1 != nullptr);
    GGML_ASSERT(act2 != nullptr && w2 != nullptr && mm2 != nullptr);
    GGML_ASSERT(act3 != nullptr && w3 != nullptr && mm3 != nullptr);
    GGML_ASSERT(act4 != nullptr && w4 != nullptr && mm4 != nullptr);
    GGML_ASSERT(rms != nullptr);
    GGML_ASSERT(sum0 != nullptr);
    GGML_ASSERT(sum != nullptr);

    ggml_set_name(mm0, "eligible_mm0");
    ggml_set_name(mm1, "ineligible_mm1");
    ggml_set_name(mm2, "eligible_mm2");
    ggml_set_name(mm3, "eligible_mm3");
    ggml_set_name(mm4, "batched_mm_should_stay_cpu");
    ggml_set_name(rms, "rms_should_stay_cpu");
    ggml_set_name(sum0, "sum0");
    ggml_set_name(sum, "sum");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    GGML_ASSERT(gf != nullptr);
    ggml_build_forward_expand(gf, mm1);
    ggml_build_forward_expand(gf, mm4);
    ggml_build_forward_expand(gf, rms);
    ggml_build_forward_expand(gf, sum);

    GGML_ASSERT(runner.eligible_f16_mul_mat(mm0));
    GGML_ASSERT(!runner.eligible_f16_mul_mat(mm1));
    GGML_ASSERT(runner.eligible_f16_mul_mat(mm2));
    GGML_ASSERT(runner.eligible_f16_mul_mat(mm3));
    // 当前 HTP row-major F16 matmul 桥接只验证过 2D act/dst，带 batch 维度的 MUL_MAT 先必须留在 CPU。
    GGML_ASSERT(!runner.eligible_f16_mul_mat(mm4));
    GGML_ASSERT(!runner.eligible_f16_mul_mat(sum0));
    GGML_ASSERT(!runner.eligible_f16_mul_mat(sum));
    GGML_ASSERT(!runner.eligible_f16_mul_mat(rms));

    // 类型/shape 支持不等于本轮 diffusion HTP target，src0 不是参数权重时不能被自动规划到 NPU。
    GGML_ASSERT(!runner.htp_target(mm0));
    GGML_ASSERT(!runner.htp_target(mm1));
    GGML_ASSERT(!runner.htp_target(mm2));
    GGML_ASSERT(!runner.htp_target(mm3));
    GGML_ASSERT(!runner.htp_target(mm4));
    // 当前阶段目标只覆盖 diffusion F16 matmul，已经验证过的 RMS_NORM 先不继续纳入这条自动路由。
    GGML_ASSERT(!runner.htp_target(rms));
    GGML_ASSERT(!runner.htp_target(sum0));
    GGML_ASSERT(!runner.htp_target(sum));

    std::vector<ggml_tensor *> targets = runner.planned_targets(gf);
    GGML_ASSERT(targets.empty());
    GGML_ASSERT(!contains_tensor(targets, mm0));
    GGML_ASSERT(!contains_tensor(targets, mm1));
    GGML_ASSERT(!contains_tensor(targets, mm2));
    GGML_ASSERT(!contains_tensor(targets, mm3));
    GGML_ASSERT(!contains_tensor(targets, mm4));
    GGML_ASSERT(!contains_tensor(targets, rms));

    GGML_ASSERT(find_node_index(gf, mm0) >= 0);
    GGML_ASSERT(find_node_index(gf, mm1) >= 0);
    GGML_ASSERT(find_node_index(gf, mm2) >= 0);
    GGML_ASSERT(find_node_index(gf, mm3) >= 0);
    GGML_ASSERT(find_node_index(gf, mm4) >= 0);

    ggml_backend_t htp = ggml_backend_init_by_name("HTP", nullptr);
    GGML_ASSERT(htp != nullptr);
    if (ggml_backend_htp_runtime_available()) {
        const char * old = std::getenv("SD_HTP_COMPUTE");
        std::string old_value = old != nullptr ? old : "";
        const bool had_old = old != nullptr;
        setenv("SD_HTP_COMPUTE", "1", 1);
        PlacementProbeRunner placement_runner(cpu, cpu, htp);
        GGML_ASSERT(placement_runner.alloc_params_buffer());
        if (had_old) {
            setenv("SD_HTP_COMPUTE", old_value.c_str(), 1);
        } else {
            unsetenv("SD_HTP_COMPUTE");
        }

        fill_tensor_f16(placement_runner.eligible_w0, 0.25f);
        fill_tensor_f32(placement_runner.ineligible_w1, 0.5f);
        fill_tensor_f16(placement_runner.eligible_w2, 0.75f);

        GGML_ASSERT(placement_runner.get_default_params_buffer() != nullptr);
        // 新的目标是参数分配阶段直接把可执行的 F16 权重放进 RPCMEM，不再等图执行前复制。
        GGML_ASSERT(placement_runner.get_direct_htp_params_buffer() != nullptr);
        GGML_ASSERT(tensor_is_rpcmem(placement_runner.eligible_w0));
        GGML_ASSERT(!tensor_is_rpcmem(placement_runner.ineligible_w1));
        GGML_ASSERT(tensor_is_rpcmem(placement_runner.eligible_w2));
        // 普通 2D F16 参数如果没有被模型结构标记为 MUL_MAT weight，不能被盲目放到 RPCMEM。
        GGML_ASSERT(!tensor_is_rpcmem(placement_runner.unmarked_w3));
        GGML_ASSERT(placement_runner.direct_htp_param_count() == 3);

        ggml_cgraph * placement_gf = placement_runner.build_graph();
        GGML_ASSERT(placement_gf != nullptr);
        std::vector<ggml_tensor *> placement_targets = placement_runner.planned_targets(placement_gf);
        GGML_ASSERT(placement_targets.size() == 2);

        GGML_ASSERT(placement_runner.get_direct_htp_params_buffer() != nullptr);
        GGML_ASSERT(tensor_is_rpcmem(placement_runner.eligible_w0));
        GGML_ASSERT(!tensor_is_rpcmem(placement_runner.ineligible_w1));
        GGML_ASSERT(tensor_is_rpcmem(placement_runner.eligible_w2));
        GGML_ASSERT(!tensor_is_rpcmem(placement_runner.unmarked_w3));
    } else {
        std::printf("[SKIP] htp persistent placement probe: runtime unavailable (%s)\n",
                    ggml_backend_htp_runtime_error() != nullptr ? ggml_backend_htp_runtime_error() : "unknown");
    }

    {
        const char * old = std::getenv("SD_HTP_COMPUTE");
        std::string old_value = old != nullptr ? old : "";
        const bool had_old = old != nullptr;
        unsetenv("SD_HTP_COMPUTE");
        PlacementProbeRunner fake_htp_runner(cpu, cpu_params, cpu);
        GGML_ASSERT(fake_htp_runner.get_runtime_backend() != fake_htp_runner.get_params_backend());
        GGML_ASSERT(fake_htp_runner.alloc_params_buffer());
        if (had_old) {
            setenv("SD_HTP_COMPUTE", old_value.c_str(), 1);
        } else {
            unsetenv("SD_HTP_COMPUTE");
        }

        fill_tensor_f16(fake_htp_runner.eligible_w0, 0.15f);
        fill_tensor_f32(fake_htp_runner.ineligible_w1, 0.35f);
        fill_tensor_f16(fake_htp_runner.eligible_w2, 0.55f);

        auto out = fake_htp_runner.run_fake_htp_compute();
        GGML_ASSERT(out.has_value());
        GGML_ASSERT(fake_htp_runner.direct_htp_param_count() == 0);
        // HTP 参数直接放置后，不应该再额外创建 partial runtime params buffer。
        GGML_ASSERT(!fake_htp_runner.has_partial_runtime_params_buffer());
        // CPU 伪装 HTP 不会产生 RPCMEM placement，因此全部参数仍走普通 runtime offload。
        GGML_ASSERT(fake_htp_runner.runtime_offload_pair_count() == 5);
    }

    if (ggml_backend_htp_runtime_available()) {
        const char * old = std::getenv("SD_HTP_COMPUTE");
        std::string old_value = old != nullptr ? old : "";
        const bool had_old = old != nullptr;
        setenv("SD_HTP_COMPUTE", "1", 1);
        PlacementProbeRunner view_runner(cpu, cpu, htp);
        GGML_ASSERT(view_runner.alloc_params_buffer());
        if (had_old) {
            setenv("SD_HTP_COMPUTE", old_value.c_str(), 1);
        } else {
            unsetenv("SD_HTP_COMPUTE");
        }

        ggml_cgraph * view_gf = view_runner.build_view_weight_graph();
        GGML_ASSERT(view_gf != nullptr);
        GGML_ASSERT(view_runner.view_root != nullptr);
        GGML_ASSERT(view_runner.eligible_view != nullptr);
        fill_tensor_f16(view_runner.view_root, 0.95f);

        std::vector<ggml_tensor *> persistent_weights = view_runner.directly_placed_htp_weights(view_gf);
        // llama.cpp-npu 式直接参数放置会把参数 root 放进 RPCMEM，view 权重通过 backing root 的 offset 进入 HTP。
        GGML_ASSERT(contains_tensor(persistent_weights, view_runner.view_root));
        std::vector<ggml_tensor *> view_targets = view_runner.planned_targets(view_gf);
        GGML_ASSERT(view_targets.size() == 1);

        int node_id_sum = 0;
        int leaf_id_sum = 0;
        view_runner.planned_buffer_id_sums(view_gf, &node_id_sum, &leaf_id_sum);
        GGML_ASSERT(node_id_sum == 2);
        GGML_ASSERT(leaf_id_sum == 1);

        GGML_ASSERT(view_runner.direct_htp_param_count() == 3);
        GGML_ASSERT(view_runner.direct_htp_contains(view_runner.view_root));
    }

    std::printf("test-backend-htp-diffusion-f16-routing passed\n");

    ggml_backend_free(htp);
    ggml_backend_free(cpu_params);
    ggml_backend_free(cpu);
    return 0;
}
