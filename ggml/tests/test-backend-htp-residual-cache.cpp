#include "ggml-backend.h"
#include "ggml-htp.h"
#include "ggml-opencl-htp.h"
#include "ggml.h"
#include "common_dit.hpp"
#include "ggml_extend.hpp"
#include "mmdit.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <vector>

namespace {

constexpr int k_skip = 77;

struct graph_order_result {
    size_t buffer_size = 0;
    int cache_count = 0;
};

bool graph_contains(ggml_cgraph * graph, const ggml_tensor * tensor) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == tensor) {
            return true;
        }
    }
    return false;
}

const ggml_tensor * backing_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

bool tensor_has_resolved_backing(const ggml_tensor * tensor) {
    const ggml_tensor * root = backing_root(tensor);
    return root != nullptr && root->buffer != nullptr && root->data != nullptr;
}

int graph_node_index(ggml_cgraph * graph, const ggml_tensor * tensor) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == tensor) {
            return i;
        }
    }
    return -1;
}

struct local_attention_pair_fixture {
    ggml_tensor * mask = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * dense = nullptr;
    ggml_tensor * sparse = nullptr;
    ggml_tensor * rho = nullptr;
    ggml_tensor * residual = nullptr;
};

ggml_tensor * graph_order_custom(
        ggml_context * ctx,
        const char * name,
        std::initializer_list<ggml_tensor *> sources) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    tensor->op = GGML_OP_CUSTOM;
    int source_index = 0;
    for (ggml_tensor * source : sources) {
        tensor->src[source_index++] = source;
    }
    ggml_set_name(tensor, name);
    return tensor;
}

local_attention_pair_fixture build_local_attention_pair(
        ggml_context * ctx,
        ggml_cgraph * graph,
        int pair_index,
        bool view_backed_dense = false) {
    local_attention_pair_fixture pair;
    ggml_tensor * q_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * k_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * v_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * mask_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    pair.mask = graph_order_custom(ctx, "local_mask", {mask_input});
    pair.q = graph_order_custom(ctx, "local_q", {q_input});
    pair.k = graph_order_custom(ctx, "local_k", {k_input});
    pair.v = graph_order_custom(ctx, "local_v", {v_input});
    ggml_tensor * dense_storage = graph_order_custom(
        ctx, "local_dense_storage", {pair.q, pair.k, pair.v});
    pair.dense = dense_storage;
    if (view_backed_dense) {
        ggml_tensor * reshaped = ggml_reshape_2d(ctx, dense_storage, 8, 8);
        pair.dense = ggml_permute(ctx, reshaped, 1, 0, 2, 3);
        ggml_set_name(pair.dense, "local_dense_view");
    } else {
        ggml_set_name(pair.dense, "local_dense");
    }
    pair.sparse = graph_order_custom(
        ctx, "local_sparse", {pair.q, pair.k, pair.v, pair.mask});
    pair.rho = graph_order_custom(ctx, "local_rho", {pair.dense, pair.sparse});
    pair.residual = graph_order_custom(
        ctx, "local_residual", {pair.dense, pair.sparse, pair.rho});

    const auto persist = [](const std::string &, ggml_tensor * tensor) {
        return tensor;
    };
    sd_cache_tensor_and_expand_local_residual_graph(
        graph, persist,
        "htp_block_sparse_residual_rho_b0_pair_s" +
            std::to_string(pair_index),
        pair.rho);
    sd_cache_tensor_and_expand_local_residual_graph(
        graph, persist,
        "htp_block_sparse_residual_b0_pair_s" +
            std::to_string(pair_index),
        pair.residual);
    return pair;
}

bool local_residual_graph_expands_pairs_in_attention_order() {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    const auto first = build_local_attention_pair(ctx, graph, 0);
    const auto second = build_local_attention_pair(ctx, graph, 1);
    ggml_tensor * continuation = graph_order_custom(
        ctx, "local_continuation", {second.dense});
    ggml_build_forward_expand(graph, continuation);

    const int first_mask = graph_node_index(graph, first.mask);
    const int first_q = graph_node_index(graph, first.q);
    const int first_k = graph_node_index(graph, first.k);
    const int first_v = graph_node_index(graph, first.v);
    const int first_dense = graph_node_index(graph, first.dense);
    const int first_sparse = graph_node_index(graph, first.sparse);
    const int first_residual = graph_node_index(graph, first.residual);
    const int second_mask = graph_node_index(graph, second.mask);
    const int second_q = graph_node_index(graph, second.q);
    const int second_residual = graph_node_index(graph, second.residual);
    const bool ordered =
        first_mask >= 0 && first_mask < first_q && first_mask < first_k &&
        first_mask < first_v && first_q < first_dense && first_k < first_dense &&
        first_v < first_dense && first_dense < first_sparse &&
        first_sparse < first_residual && first_residual < second_mask &&
        second_mask < second_q && second_q < second_residual;
    const bool local_outputs =
        (first.rho->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
        (first.residual->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
        (second.rho->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
        (second.residual->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    ggml_free(ctx);
    return ordered && local_outputs;
}

enum class no_alloc_role {
    sparse,
    dense,
    rho,
    residual,
};

struct allocator_pair_contract_result {
    bool allocated = false;
    bool dense_backed = false;
    bool sparse_backed = false;
    bool rho_backed = false;
    bool residual_backed = false;
    bool continuation_backed = false;
    bool sparse_collected = false;
    size_t buffer_size = SIZE_MAX;
    size_t sparse_size = 0;
};

allocator_pair_contract_result run_allocator_pair_contract(
        bool view_backed_dense,
        no_alloc_role role,
        bool add_sparse_consumer = false,
        bool break_pair = false) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    allocator_pair_contract_result result;
    if (ctx == nullptr || backend == nullptr) {
        if (backend != nullptr) ggml_backend_free(backend);
        if (ctx != nullptr) ggml_free(ctx);
        return result;
    }

    ggml_tensor * dense_storage = graph_order_custom(
        ctx, "no_alloc_dense_storage", {});
    ggml_tensor * dense = dense_storage;
    if (view_backed_dense) {
        ggml_tensor * reshaped = ggml_reshape_2d(ctx, dense_storage, 8, 8);
        dense = ggml_permute(ctx, reshaped, 1, 0, 2, 3);
        ggml_set_name(dense, "no_alloc_dense_view");
    } else {
        ggml_set_name(dense, "no_alloc_dense");
    }
    ggml_tensor * sparse = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, 1024 * 1024);
    sparse->op = GGML_OP_CUSTOM;
    ggml_set_name(sparse, "no_alloc_sparse_destination");
    ggml_tensor * rho = graph_order_custom(
        ctx, "htp_block_sparse_residual_rho_b0_allocator_s0",
        {dense, sparse});
    ggml_tensor * residual = graph_order_custom(
        ctx, "htp_block_sparse_residual_b0_allocator_s0",
        {dense, sparse, rho});
    if (break_pair) {
        rho->src[1] = dense;
    }
    ggml_tensor * continuation = graph_order_custom(
        ctx, "no_alloc_dense_continuation", {dense, residual});
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, continuation);
    if (add_sparse_consumer) {
        ggml_tensor * sparse_consumer = graph_order_custom(
            ctx, "no_alloc_sparse_consumer", {sparse});
        ggml_build_forward_expand(graph, sparse_consumer);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const ggml_tensor * no_alloc_tensor = sparse;
    switch (role) {
        case no_alloc_role::sparse:   no_alloc_tensor = sparse;   break;
        case no_alloc_role::dense:    no_alloc_tensor = dense;    break;
        case no_alloc_role::rho:      no_alloc_tensor = rho;      break;
        case no_alloc_role::residual: no_alloc_tensor = residual; break;
    }
    const ggml_tensor * no_alloc[] = {no_alloc_tensor};
    const bool configured = ggml_gallocr_set_no_alloc_tensors(
        allocator, no_alloc, 1);
    result.allocated = configured &&
        ggml_gallocr_reserve(allocator, graph) &&
        ggml_gallocr_alloc_graph(allocator, graph);
    result.buffer_size = result.allocated ?
        ggml_gallocr_get_buffer_size(allocator, 0) : SIZE_MAX;
    result.sparse_size = ggml_nbytes(sparse);
    result.dense_backed = tensor_has_resolved_backing(dense);
    result.sparse_backed = tensor_has_resolved_backing(sparse);
    result.rho_backed = tensor_has_resolved_backing(rho);
    result.residual_backed = tensor_has_resolved_backing(residual);
    result.continuation_backed = tensor_has_resolved_backing(continuation);
    const auto collected =
        sd_collect_local_residual_owned_sparse_producers(graph);
    result.sparse_collected =
        collected.size() == 1 && collected.front() == sparse;

    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

bool no_allocation_tensor_avoids_sparse_destination_storage() {
    const allocator_pair_contract_result direct =
        run_allocator_pair_contract(false, no_alloc_role::sparse);
    const allocator_pair_contract_result view =
        run_allocator_pair_contract(true, no_alloc_role::sparse);
    return direct.allocated && direct.dense_backed && !direct.sparse_backed &&
        direct.rho_backed && direct.residual_backed &&
        direct.continuation_backed && direct.sparse_collected &&
        direct.buffer_size < direct.sparse_size &&
        view.allocated && view.dense_backed && !view.sparse_backed &&
        view.rho_backed && view.residual_backed &&
        view.continuation_backed && view.sparse_collected &&
        view.buffer_size < view.sparse_size;
}

bool no_allocation_scope_is_limited_to_local_sparse_output() {
    const allocator_pair_contract_result dense =
        run_allocator_pair_contract(false, no_alloc_role::dense);
    const allocator_pair_contract_result rho =
        run_allocator_pair_contract(false, no_alloc_role::rho);
    const allocator_pair_contract_result residual =
        run_allocator_pair_contract(false, no_alloc_role::residual);
    const allocator_pair_contract_result ordinary_consumer =
        run_allocator_pair_contract(
            false, no_alloc_role::sparse, true, false);
    const allocator_pair_contract_result broken_pair =
        run_allocator_pair_contract(
            false, no_alloc_role::sparse, false, true);
    return dense.allocated && !dense.dense_backed && dense.sparse_backed &&
        dense.rho_backed && dense.residual_backed &&
        rho.allocated && rho.dense_backed && rho.sparse_backed &&
        !rho.rho_backed && rho.residual_backed &&
        residual.allocated && residual.dense_backed && residual.sparse_backed &&
        residual.rho_backed && !residual.residual_backed &&
        ordinary_consumer.allocated && !ordinary_consumer.sparse_backed &&
        !ordinary_consumer.sparse_collected &&
        broken_pair.allocated && !broken_pair.sparse_backed &&
        !broken_pair.sparse_collected;
}

bool allocation_record_collector_preserves_pair_roles() {
    auto run_case = [](int pair_count) {
        ggml_init_params params = {
            /*.mem_size   =*/ 8 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (ctx == nullptr || backend == nullptr) {
            if (backend != nullptr) ggml_backend_free(backend);
            if (ctx != nullptr) ggml_free(ctx);
            return false;
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
        std::vector<local_attention_pair_fixture> pairs;
        pairs.reserve(pair_count);
        for (int i = 0; i < pair_count; ++i) {
            pairs.push_back(build_local_attention_pair(
                ctx, graph, i, (i % 2) != 0));
            const std::string rho_name =
                "htp_block_sparse_residual_rho_b0_records_s" +
                std::to_string(i);
            const std::string residual_name =
                "htp_block_sparse_residual_b0_records_s" +
                std::to_string(i);
            ggml_set_name(pairs.back().rho, rho_name.c_str());
            ggml_set_name(pairs.back().residual, residual_name.c_str());
        }
        ggml_tensor * continuation = graph_order_custom(
            ctx, "allocation_record_continuation",
            {pairs.back().dense, pairs.back().residual});
        ggml_build_forward_expand(graph, continuation);
        if (pair_count >= 2) {
            ggml_tensor * ordinary_consumer = graph_order_custom(
                ctx, "allocation_record_sparse_consumer",
                {pairs.back().sparse});
            ggml_build_forward_expand(graph, ordinary_consumer);
        }

        ggml_backend_buffer_t persistent_buffer =
            ggml_backend_alloc_buffer(backend, 1024 * 1024);
        if (persistent_buffer == nullptr) {
            ggml_backend_free(backend);
            ggml_free(ctx);
            return false;
        }
        const size_t alignment = ggml_backend_get_alignment(backend);
        size_t persistent_offset = 0;
        for (const local_attention_pair_fixture & pair : pairs) {
            for (ggml_tensor * tensor : {pair.rho, pair.residual}) {
                persistent_offset =
                    (persistent_offset + alignment - 1) / alignment * alignment;
                if (ggml_backend_tensor_alloc(
                        persistent_buffer, tensor,
                        static_cast<char *>(
                            ggml_backend_buffer_get_base(persistent_buffer)) +
                            persistent_offset) != GGML_STATUS_SUCCESS) {
                    ggml_backend_buffer_free(persistent_buffer);
                    ggml_backend_free(backend);
                    ggml_free(ctx);
                    return false;
                }
                persistent_offset += ggml_nbytes(tensor);
            }
        }

        std::vector<const ggml_tensor *> no_alloc_tensors;
        for (int i = 0; i < pair_count; ++i) {
            if ((i % 2) == 0) {
                no_alloc_tensors.push_back(pairs[i].sparse);
            }
        }
        ggml_gallocr_t allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        const bool allocated =
            ggml_gallocr_set_no_alloc_tensors(
                allocator,
                no_alloc_tensors.empty() ? nullptr : no_alloc_tensors.data(),
                no_alloc_tensors.size()) &&
            ggml_gallocr_reserve(allocator, graph) &&
            ggml_gallocr_alloc_graph(allocator, graph);

        std::vector<int> node_buffer_ids(ggml_graph_n_nodes(graph), 0);
        std::vector<int> leaf_buffer_ids(
            sd::ggml_graph_cut::leaf_count(graph), 0);
        for (int i = 0; i < pair_count; ++i) {
            const int dense_index = graph_node_index(graph, pairs[i].dense);
            const int sparse_index = graph_node_index(graph, pairs[i].sparse);
            if (dense_index >= 0) node_buffer_ids[dense_index] = 1;
            if (sparse_index >= 0) node_buffer_ids[sparse_index] = 2;
        }
        const std::unordered_set<ggml_backend_buffer_t> persistent_backings = {
            persistent_buffer,
        };
        const auto records =
            sd_collect_htp_residual_pair_allocation_records(
                graph, no_alloc_tensors, node_buffer_ids, leaf_buffer_ids,
                ggml_backend_get_default_buffer_type(backend), nullptr, nullptr,
                persistent_backings);
        const auto summary =
            sd_summarize_htp_residual_pair_allocation_records(records);

        bool valid = allocated &&
            records.size() == static_cast<size_t>(4 * pair_count) &&
            summary.dense_outputs_expected == static_cast<size_t>(pair_count) &&
            summary.dense_outputs_with_backing == static_cast<size_t>(pair_count) &&
            summary.dense_outputs_no_alloc == 0 &&
            summary.metadata_only_sparse_without_backing ==
                static_cast<size_t>((pair_count + 1) / 2) &&
            summary.allocated_sparse_with_backing ==
                static_cast<size_t>(pair_count / 2) &&
            summary.persistent_outputs_expected ==
                static_cast<size_t>(2 * pair_count) &&
            summary.persistent_outputs_with_backing ==
                static_cast<size_t>(2 * pair_count);
        for (int i = 0; i < pair_count && valid; ++i) {
            const auto & dense = records[4 * i + 0];
            const auto & sparse = records[4 * i + 1];
            const auto & rho = records[4 * i + 2];
            const auto & residual = records[4 * i + 3];
            valid = dense.pair_id == static_cast<size_t>(i + 1) &&
                dense.role == SD_HTP_RESIDUAL_PAIR_DENSE &&
                dense.tensor == pairs[i].dense && dense.resolved_backing != nullptr &&
                sparse.role == SD_HTP_RESIDUAL_PAIR_SPARSE &&
                sparse.tensor == pairs[i].sparse &&
                sparse.planned_buffer_id == 2 &&
                sparse.no_alloc_direct == ((i % 2) == 0) &&
                (i % 2 == 0 ? sparse.resolved_backing == nullptr
                            : sparse.resolved_backing != nullptr) &&
                rho.role == SD_HTP_RESIDUAL_PAIR_RHO &&
                rho.tensor == pairs[i].rho &&
                rho.actual_buffer_class == SD_HTP_BUFFER_PERSISTENT &&
                residual.role == SD_HTP_RESIDUAL_PAIR_RESIDUAL &&
                residual.tensor == pairs[i].residual &&
                residual.actual_buffer_class == SD_HTP_BUFFER_PERSISTENT;
            if (i == pair_count - 1 && pair_count >= 2) {
                valid = valid && sparse.ordinary_consumer_count == 1;
            }
            if ((i % 2) != 0) {
                valid = valid && dense.view_root != dense.tensor &&
                    dense.root_planned_buffer_id >= 0;
            }
        }
        if (!valid) {
            char summary_line[512] = {};
            sd_format_htp_residual_pair_allocation_summary(
                summary_line, sizeof(summary_line), "host_fixture", summary);
            std::fprintf(stderr, "pair_count=%d %s\n", pair_count, summary_line);
            for (const auto & record : records) {
                char record_line[1024] = {};
                sd_format_htp_residual_pair_allocation_record(
                    record_line, sizeof(record_line), "host_fixture", record);
                std::fprintf(stderr, "%s\n", record_line);
            }
        }

        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(persistent_buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return valid;
    };

    return run_case(1) && run_case(2) && run_case(8);
}

bool local_residual_owned_sparse_requires_exact_pair_without_other_consumers() {
    auto build_case = [](bool add_consumer, bool break_pair) {
        ggml_init_params params = {
            /*.mem_size   =*/ 2 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }

        ggml_tensor * dense = graph_order_custom(ctx, "local_owned_dense", {});
        ggml_tensor * sparse = graph_order_custom(ctx, "local_owned_sparse", {});
        ggml_tensor * rho = graph_order_custom(ctx, "local_owned_rho", {dense, sparse});
        ggml_set_name(rho, "htp_block_sparse_residual_rho_b0_pair_s0");
        ggml_tensor * residual = graph_order_custom(
            ctx, "local_owned_residual", {dense, sparse, rho});
        ggml_set_name(residual, "htp_block_sparse_residual_b0_pair_s0");
        if (break_pair) {
            rho->src[1] = dense;
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, residual);
        if (add_consumer) {
            ggml_tensor * consumer = graph_order_custom(
                ctx, "local_owned_ordinary_consumer", {sparse});
            ggml_build_forward_expand(graph, consumer);
        }

        const auto producers =
            sd_collect_local_residual_owned_sparse_producers(graph);
        const bool result = producers.size() ==
                (add_consumer || break_pair ? 0u : 1u) &&
            (producers.empty() || producers.front() == sparse);
        ggml_free(ctx);
        return result;
    };

    return build_case(false, false) &&
        build_case(true, false) &&
        build_case(false, true);
}

bool local_lifetime_pair_state_and_pool_reuse_are_monotonic() {
    ggml_htp_opencl_pair_lifetime_state pair = {};
    if (!ggml_htp_opencl_pair_mark_sparse_committed(&pair) ||
        !ggml_htp_opencl_pair_mark_residual_submitted(&pair) ||
        !ggml_htp_opencl_pair_mark_sparse_complete(&pair) ||
        !ggml_htp_opencl_pair_mark_sparse_complete(&pair) ||
        !ggml_htp_opencl_pair_mark_residual_complete(&pair) ||
        !ggml_htp_opencl_pair_mark_reaped(&pair) ||
        ggml_htp_opencl_pair_mark_residual_submitted(&pair)) {
        return false;
    }

    size_t capacities[2] = {};
    bool in_use[2] = {};
    size_t slot_count = 0;
    bool resize = false;
    int slot = ggml_htp_opencl_staging_pool_select_slot(
        capacities, in_use, slot_count, 2, 1024, &resize);
    if (slot != 0 || !resize) {
        return false;
    }
    slot_count = 1;
    capacities[0] = 1024;
    in_use[0] = true;

    slot = ggml_htp_opencl_staging_pool_select_slot(
        capacities, in_use, slot_count, 2, 512, &resize);
    if (slot != 1 || !resize) {
        return false;
    }
    slot_count = 2;
    capacities[1] = 512;
    in_use[1] = true;

    slot = ggml_htp_opencl_staging_pool_select_slot(
        capacities, in_use, slot_count, 2, 256, &resize);
    if (slot != -1) {
        return false;
    }

    in_use[0] = false;
    slot = ggml_htp_opencl_staging_pool_select_slot(
        capacities, in_use, slot_count, 2, 768, &resize);
    if (slot != 0 || resize) {
        return false;
    }
    in_use[0] = true;
    in_use[1] = false;
    slot = ggml_htp_opencl_staging_pool_select_slot(
        capacities, in_use, slot_count, 2, 2048, &resize);
    return slot == 1 && resize;
}

bool fused_pair_state_skips_sparse_event() {
    ggml_htp_opencl_pair_lifetime_state pair = {};
    if (!ggml_htp_opencl_pair_mark_fused_anchor_produced(&pair)) {
        return false;
    }
    if (!pair.sparse_committed || !pair.sparse_complete ||
        pair.residual_submitted || pair.residual_complete || pair.reaped) {
        return false;
    }
    if (!ggml_htp_opencl_pair_mark_residual_submitted(&pair) ||
        !ggml_htp_opencl_pair_mark_residual_complete(&pair) ||
        !ggml_htp_opencl_pair_mark_reaped(&pair)) {
        return false;
    }
    return pair.reaped;
}

bool staging_pressure_waits_for_residual_publication_before_cache_wait() {
    ggml_htp_opencl_pair_lifetime_state pair = {};
    if (!ggml_htp_opencl_pair_mark_sparse_committed(&pair) ||
        !ggml_htp_opencl_pair_mark_sparse_complete(&pair)) {
        return false;
    }

    return ggml_htp_opencl_pair_reap_requires_residual_submission_wait(
               &pair, true) &&
        !ggml_htp_opencl_pair_reap_requires_residual_submission_wait(
               &pair, false) &&
        ggml_htp_opencl_pair_mark_residual_submitted(&pair) &&
        !ggml_htp_opencl_pair_reap_requires_residual_submission_wait(
               &pair, true);
}

bool local_memory_plan_reports_allocator_ownership() {
    const sd_htp_opencl_memory_plan plan = sd_make_htp_opencl_memory_plan(
        GGML_HTP_OPENCL_LIFETIME_REUSE,
        1024,
        2048,
        4096,
        8192,
        16384,
        20480,
        37,
        37,
        74,
        true,
        0);
    char summary[512] = {};
    const int written = sd_format_htp_opencl_memory_plan(
        summary, sizeof(summary), plan);
    return written > 0 &&
        std::strcmp(
            summary,
            "htp_opencl_memory_plan mode=reuse "
            "temporary_runtime_bytes=1024 temporary_htp_bytes=2048 "
            "temporary_qkv_bytes=4096 host_cache_bytes=8192 "
            "rpcmem_cache_logical_bytes=16384 "
            "rpcmem_cache_allocation_bytes=20480 "
            "sparse_destinations_in_gallocr=0 "
            "cache_outputs_in_gallocr=0 retained_roots=0") == 0;
}

bool cache_expands_persisted_tensor() {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    ggml_tensor * view = ggml_transpose(ctx, source);
    ggml_tensor * persisted = nullptr;
    const std::function<ggml_tensor *(const std::string &, ggml_tensor *)>
        cache_tensor = [&](const std::string &, ggml_tensor * tensor) {
            persisted = ggml_cont(ctx, tensor);
            ggml_set_name(persisted, "graph_order_persisted_cache");
            return persisted;
        };

    sd_cache_tensor_and_expand_graph(graph, cache_tensor, "cache", view);
    const bool expanded = persisted != nullptr && graph_contains(graph, persisted);
    ggml_free(ctx);
    return expanded;
}

bool opencl_residual_route_requires_all_prerequisites() {
    unsetenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR");
    unsetenv("SD_HTP_SPARSE_OPENCL_RESIDUAL");
    unsetenv("SD_HTP_SPARSE_OPENCL");
    unsetenv("SD_HTP_SPARSE_OPENCL_ASYNC");
    unsetenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT");
    unsetenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE");
    unsetenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO");
    if (sd_htp_opencl_residual_route_prerequisites()) {
        return false;
    }

    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    if (sd_htp_opencl_residual_route_prerequisites()) {
        return false;
    }

    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    return sd_htp_opencl_residual_route_prerequisites();
}

bool fused_residual_only_route_avoids_sparse_async_prerequisites() {
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    unsetenv("SD_HTP_SPARSE_OPENCL_ASYNC");
    unsetenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT");
    unsetenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD");
    return sd_htp_opencl_residual_route_prerequisites() &&
           !sd_htp_opencl_residual_route_has_deferred_job();
}

bool cache_storage_updates_avoid_rebuilding_disjoint_branches() {
    const std::vector<std::unordered_set<std::string>> existing = {
        {"residual_branch_0", "rho_branch_0"},
    };

    const auto append = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_1", "rho_branch_1"},
        false);
    if (append.mode != SD_CACHE_STORAGE_APPEND ||
        !append.replace_segment_indices.empty()) {
        return false;
    }

    const auto replace = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_0", "rho_branch_0"},
        false);
    if (replace.mode != SD_CACHE_STORAGE_REPLACE_SEGMENTS ||
        replace.replace_segment_indices != std::vector<size_t>{0}) {
        return false;
    }

    const auto partial = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_0"},
        false);
    if (partial.mode != SD_CACHE_STORAGE_MERGE) {
        return false;
    }

    const auto graph_cut = sd_plan_cache_storage_update(
        existing,
        {"residual_branch_1", "rho_branch_1"},
        true);
    return graph_cut.mode == SD_CACHE_STORAGE_MERGE;
}

graph_order_result build_graph_order_fixture(bool expand_cache_immediately) {
    constexpr int k_layers = 8;
    constexpr int64_t k_elements = 256 * 1024;

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return {};
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        ggml_free(ctx);
        return {};
    }
    ggml_gallocr_t allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4 * k_layers + 16, false);

    graph_order_result result;
    std::vector<ggml_tensor *> cache_outputs;
    ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_elements);
    ggml_set_name(state, "graph_order_input");

    const auto cache_tensor = [&](const std::string &, ggml_tensor * tensor) {
        ++result.cache_count;
        cache_outputs.push_back(tensor);
        return tensor;
    };

    for (int layer = 0; layer < k_layers; ++layer) {
        ggml_tensor * branch = ggml_sqr(ctx, state);
        ggml_tensor * cache = ggml_sum(ctx, branch);
        ggml_format_name(cache, "graph_order_cache_%d", layer);
        if (expand_cache_immediately) {
            sd_cache_tensor_and_expand_graph(graph, cache_tensor, cache->name, cache);
        } else {
            cache_tensor(cache->name, cache);
        }
        state = ggml_neg(ctx, state);
        ggml_format_name(state, "graph_order_main_%d", layer);
    }

    ggml_build_forward_expand(graph, state);
    if (!expand_cache_immediately) {
        for (ggml_tensor * cache : cache_outputs) {
            ggml_set_output(cache);
            ggml_build_forward_expand(graph, cache);
        }
    }

    if (ggml_gallocr_reserve(allocator, graph)) {
        result.buffer_size = ggml_gallocr_get_buffer_size(allocator, 0);
    }

    ggml_gallocr_free(allocator);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

enum class cache_input_failure_case {
    none,
    first_reserve,
    second_alloc,
    preflight,
    compute,
};

struct cache_input_reserve_result {
    bool executed = false;
    int phase = 0;
    size_t runtime_bytes = 0;
    size_t cache_bytes = 0;
    bool failure_observed = false;
    bool tensor_restored = false;
    bool consumers_released = false;
};

#if defined(GGML_USE_HTP)

class cache_input_reserve_runner final : public GGMLRunner {
public:
    cache_input_reserve_runner(ggml_backend_t runtime, ggml_backend_t params)
        : GGMLRunner(runtime, params) {}

    std::string get_desc() override {
        return "cache-input-reserve-fixture";
    }

    void adopt_cache_segment(
            ggml_context * ctx,
            ggml_backend_buffer_t buffer,
            const std::vector<std::pair<std::string, ggml_tensor *>> & tensors) {
        CacheStorageSegment segment;
        segment.ctx = ctx;
        segment.buffer = buffer;
        for (const auto & tensor : tensors) {
            segment.names.insert(tensor.first);
        }
        cache_storage_segments_.push_back(std::move(segment));
    }

    bool execute_apply(ggml_cgraph * graph, int * phase) {
        if (phase != nullptr) *phase = 1;
        CacheInputBindingTransaction transaction;
        if (!allocate_compute_graph_with_cache_inputs(graph, &transaction)) {
            rollback_cache_input_bindings(&transaction);
            return false;
        }
        if (phase != nullptr) *phase = 3;
        rollback_cache_input_bindings(&transaction);
        if (phase != nullptr) *phase = 4;
        return true;
    }

    bool execute_failure(
            ggml_cgraph * graph,
            ggml_tensor * input,
            cache_input_failure_case failure_case,
            cache_input_reserve_result * result) {
        if (graph == nullptr || input == nullptr || result == nullptr ||
            failure_case == cache_input_failure_case::none) {
            return false;
        }
        const ggml_backend_buffer_t original_buffer = input->buffer;
        void * const original_data = input->data;
        void * const original_extra = input->extra;
        const ggml_op original_op = input->op;
        ggml_tensor * original_src[GGML_MAX_SRC] = {};
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            original_src[i] = input->src[i];
        }

        std::string mock_key;
        {
            CacheInputBindingTransaction transaction;
            auto rollback = std::unique_ptr<
                CacheInputBindingTransaction,
                std::function<void(CacheInputBindingTransaction *)>>(
                    &transaction,
                    [this](CacheInputBindingTransaction * active) {
                        rollback_cache_input_bindings(active);
                    });
            CacheInputAllocationFailurePoint allocation_failure =
                CacheInputAllocationFailurePoint::none;
            if (failure_case == cache_input_failure_case::first_reserve) {
                allocation_failure =
                    CacheInputAllocationFailurePoint::first_reserve;
            } else if (failure_case ==
                       cache_input_failure_case::second_alloc) {
                allocation_failure =
                    CacheInputAllocationFailurePoint::second_alloc;
            }
            const bool allocated = allocate_compute_graph_with_cache_inputs(
                graph, &transaction, allocation_failure);
            const bool allocation_failure_expected =
                failure_case == cache_input_failure_case::first_reserve ||
                failure_case == cache_input_failure_case::second_alloc;
            result->failure_observed = allocation_failure_expected
                ? !allocated : allocated;
            if (!result->failure_observed) {
                return false;
            }

            const auto key = cache_backed_tensor_name_map.find(input);
            if (!transaction.active || key == cache_backed_tensor_name_map.end()) {
                return false;
            }
            mock_key = key->second;
            async_residual_cache_slots_[mock_key] = {
                999, nullptr, 1, false,
            };
            active_async_cache_consumers_.push_back({mock_key, 999});

            if (failure_case == cache_input_failure_case::compute) {
                finalize_cache_input_bindings(
                    &transaction, GGML_STATUS_FAILED);
            }
        }

        result->tensor_restored =
            input->buffer == original_buffer &&
            input->data == original_data &&
            input->extra == original_extra &&
            input->op == original_op;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            result->tensor_restored = result->tensor_restored &&
                input->src[i] == original_src[i];
        }
        const auto mock = async_residual_cache_slots_.find(mock_key);
        result->consumers_released =
            active_async_cache_consumers_.empty() &&
            mock != async_residual_cache_slots_.end() &&
            mock->second.consumer_count == 0;
        if (mock != async_residual_cache_slots_.end()) {
            async_residual_cache_slots_.erase(mock);
        }
        result->executed = true;
        return true;
    }

    size_t runtime_buffer_bytes() const {
        return compute_allocr != nullptr
            ? ggml_gallocr_get_buffer_size(compute_allocr, 0)
            : 0;
    }
};

cache_input_reserve_result build_cache_input_reserve_fixture(
        int pair_count,
        cache_input_failure_case failure_case =
            cache_input_failure_case::none) {
    constexpr int64_t k_cache_elements = 256 * 1024;
    cache_input_reserve_result result;
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return result;
    }

    ggml_init_params cache_params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * cache_ctx = ggml_init(cache_params);
    if (cache_ctx == nullptr) {
        ggml_backend_free(backend);
        return result;
    }

    std::vector<std::pair<std::string, ggml_tensor *>> cached;
    cached.reserve(static_cast<size_t>(2 * pair_count));
    for (int pair = 0; pair < pair_count; ++pair) {
        for (const char * role : {"residual", "rho"}) {
            const std::string name =
                "htp_block_sparse_residual_" + std::string(role) +
                "_b0_allocator_s" + std::to_string(pair);
            ggml_tensor * tensor = ggml_new_tensor_1d(
                cache_ctx, GGML_TYPE_F32, k_cache_elements);
            ggml_set_name(tensor, name.c_str());
            result.cache_bytes += ggml_nbytes(tensor);
            cached.push_back({name, tensor});
        }
    }
    ggml_backend_buffer_t cache_buffer =
        ggml_backend_alloc_ctx_tensors(cache_ctx, backend);
    if (cache_buffer == nullptr) {
        ggml_free(cache_ctx);
        ggml_backend_free(backend);
        return result;
    }
    ggml_backend_buffer_clear(cache_buffer, 0);

    {
        cache_input_reserve_runner runner(backend, backend);
        runner.adopt_cache_segment(cache_ctx, cache_buffer, cached);
        cache_ctx = nullptr;
        cache_buffer = nullptr;
        runner.reset_compute_ctx();
        GGMLRunnerContext runner_ctx = runner.get_context();
        ggml_cgraph * graph = ggml_new_graph_custom(
            runner_ctx.ggml_ctx, 4 * pair_count + 16, false);
        ggml_tensor * state = nullptr;
        ggml_tensor * first_input = nullptr;
        for (const auto & entry : cached) {
            ggml_tensor * input =
                runner_ctx.cache_tensor_as_graph_input(entry.first);
            if (input == nullptr) {
                state = nullptr;
                break;
            }
            if (first_input == nullptr) {
                first_input = input;
            }
            ggml_tensor * sum = ggml_sum(runner_ctx.ggml_ctx, input);
            state = state == nullptr
                ? sum
                : ggml_add(runner_ctx.ggml_ctx, state, sum);
        }
        if (state != nullptr) {
            ggml_build_forward_expand(graph, state);
            if (failure_case == cache_input_failure_case::none) {
                result.executed = runner.execute_apply(graph, &result.phase);
            } else {
                result.executed = runner.execute_failure(
                    graph, first_input, failure_case, &result);
            }
            result.runtime_bytes = runner.runtime_buffer_bytes();
        }
    }

    if (cache_buffer != nullptr) ggml_backend_buffer_free(cache_buffer);
    if (cache_ctx != nullptr) ggml_free(cache_ctx);
    ggml_backend_free(backend);
    return result;
}

#endif

struct packed_view_reserve_result {
    bool reserved = false;
    bool all_views = true;
    bool offsets_ok = true;
    int storage_count = 0;
    size_t buffer_bytes = 0;
    size_t storage_bytes = 0;
};

packed_view_reserve_result build_packed_view_reserve_fixture(int layer_count) {
    constexpr int64_t k_num_heads = 24;
    constexpr int64_t k_head_dim = 64;
    constexpr int64_t k_hidden = k_num_heads * k_head_dim;
    constexpr int64_t k_tokens = 64;
    packed_view_reserve_result result;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        return result;
    }
    ggml_init_params param_params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * param_ctx = ggml_init(param_params);
    ggml_init_params graph_params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * graph_ctx = ggml_init(graph_params);
    if (param_ctx == nullptr || graph_ctx == nullptr) {
        if (graph_ctx != nullptr) ggml_free(graph_ctx);
        if (param_ctx != nullptr) ggml_free(param_ctx);
        ggml_backend_free(backend);
        return result;
    }

    String2TensorStorage storage_map;
    TensorStorage weight_storage;
    weight_storage.expected_type = GGML_TYPE_F16;
    weight_storage.layout = TensorDataLayout::HTP_PERMUTED_W16A32;
    storage_map["fixture.qkv.weight"] = weight_storage;
    auto attention = std::make_shared<SelfAttention>(
        k_hidden, k_num_heads, "rms", false, false);
    attention->init(param_ctx, storage_map, "fixture");
    ggml_backend_buffer_t param_buffer =
        ggml_backend_alloc_ctx_tensors(param_ctx, backend);
    if (param_buffer == nullptr) {
        ggml_free(graph_ctx);
        ggml_free(param_ctx);
        ggml_backend_free(backend);
        return result;
    }

    setenv("SD_HTP_PACKED_ATTENTION", "1", 1);
    setenv("SD_HTP_COMPUTE", "1", 1);
    GGMLRunnerContext runner_ctx;
    runner_ctx.backend = backend;
    runner_ctx.ggml_ctx = graph_ctx;
    runner_ctx.block_sparse_attn_enabled = true;
    runner_ctx.block_sparse_residual_mode = SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY;

    ggml_tensor * input = ggml_new_tensor_3d(
        graph_ctx, GGML_TYPE_F32, k_hidden, k_tokens, 1);
    ggml_cgraph * graph = ggml_new_graph_custom(
        graph_ctx, 8 * layer_count + 16, false);
    ggml_tensor * state = nullptr;
    for (int layer = 0; layer < layer_count; ++layer) {
        PackedJointQkv packed =
            mmdit_make_packed_single_qkv(&runner_ctx, attention, input);
        if (packed.storage == nullptr || packed.qkv.size() != 3) {
            result.all_views = false;
            break;
        }
        ++result.storage_count;
        ggml_format_name(packed.storage, "packed_view_storage_%d", layer);
        const size_t qkv_bytes = ggml_nbytes(packed.storage) / 3;
        result.storage_bytes = ggml_nbytes(packed.storage);
        for (size_t qkv = 0; qkv < packed.qkv.size(); ++qkv) {
            result.all_views = result.all_views &&
                packed.qkv[qkv]->op == GGML_OP_VIEW &&
                packed.qkv[qkv]->view_src == packed.storage;
            result.offsets_ok = result.offsets_ok &&
                packed.qkv[qkv]->view_offs == qkv * qkv_bytes;
        }
        ggml_tensor * q_use = graph_order_custom(
            graph_ctx, "packed_q_use", {packed.qkv[0]});
        ggml_tensor * k_use = graph_order_custom(
            graph_ctx, "packed_k_use", {packed.qkv[1]});
        ggml_tensor * v_use = graph_order_custom(
            graph_ctx, "packed_v_use", {packed.qkv[2]});
        state = state == nullptr
            ? graph_order_custom(
                  graph_ctx, "packed_layer_state", {q_use, k_use, v_use})
            : graph_order_custom(
                  graph_ctx, "packed_layer_state", {state, q_use, k_use, v_use});
    }

    if (state != nullptr) {
        ggml_build_forward_expand(graph, state);
        ggml_gallocr_t allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        result.reserved = ggml_gallocr_reserve(allocator, graph);
        if (result.reserved) {
            result.buffer_bytes = ggml_gallocr_get_buffer_size(allocator, 0);
        }
        ggml_gallocr_free(allocator);
    }

    unsetenv("SD_HTP_COMPUTE");
    unsetenv("SD_HTP_PACKED_ATTENTION");
    ggml_backend_buffer_free(param_buffer);
    ggml_free(graph_ctx);
    ggml_free(param_ctx);
    ggml_backend_free(backend);
    return result;
}

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

int run_fp16_test() {
    const std::vector<float> dense = {
        0.0f, 1.0f, -1.0f, 0.0001220703125f,
        3.1415927f, -2.7182817f, 12.5f, -15.75f,
        0.33333334f, -0.66666669f, 102.0f, -96.0f,
    };
    const std::vector<float> sparse = {
        0.0f, 0.5f, -0.5f, 0.00006103515625f,
        2.75f, -3.0f, 10.0f, -12.0f,
        0.25f, -0.75f, 100.0f, -100.0f,
    };
    const std::vector<float> rho = {
        1.0f, 0.9999f, 1.0001f, 0.5f,
        0.125f, 0.875f, 1.25f, 0.75f,
        1.3333333f, 0.6666667f, 1.019f, 0.959f,
    };

    std::vector<float> reference(dense.size());
    std::vector<ggml_fp16_t> cached(dense.size());
    std::vector<float> recovered(dense.size());
    double squared_error = 0.0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < dense.size(); ++i) {
        reference[i] = dense[i] - sparse[i] * rho[i];
        cached[i] = ggml_fp32_to_fp16(reference[i]);
        const float residual_f32 = ggml_fp16_to_fp32(cached[i]);
        recovered[i] = sparse[i] * rho[i] + residual_f32;
        if (!check(std::isfinite(residual_f32), "FP16 residual must be finite") ||
            !check(std::isfinite(recovered[i]), "recombined output must be finite")) {
            return 1;
        }
        const float error = std::fabs(residual_f32 - reference[i]);
        max_abs = std::max(max_abs, error);
        squared_error += static_cast<double>(error) * error;
    }

    const float rmse = static_cast<float>(
        std::sqrt(squared_error / static_cast<double>(reference.size())));
    std::printf(
        "fp16 residual_type=%s rho_type=%s max_abs=%.9g rmse=%.9g nan=0 inf=0\n",
        ggml_type_name(GGML_TYPE_F16), ggml_type_name(GGML_TYPE_F32), max_abs, rmse);
    if (!check(max_abs <= 5e-3f, "FP16 residual max_abs exceeds 5e-3") ||
        !check(rmse <= 1e-3f, "FP16 residual RMSE exceeds 1e-3")) {
        return 1;
    }
    std::puts("PASS fp16");
    return 0;
}

int run_graph_order_test() {
    const graph_order_result tail = build_graph_order_fixture(false);
    const graph_order_result immediate = build_graph_order_fixture(true);
    std::printf("graph_order tail_bytes=%zu immediate_bytes=%zu caches=%d\n",
                tail.buffer_size, immediate.buffer_size, immediate.cache_count);
    if (!check(tail.buffer_size > 0 && immediate.buffer_size > 0,
               "graph-order allocator fixture must reserve both graphs") ||
        !check(tail.cache_count == 8 && immediate.cache_count == 8,
               "graph-order fixture must register every cache output") ||
        !check(cache_expands_persisted_tensor(),
               "cache expansion must use the tensor returned by persistence") ||
        !check(opencl_residual_route_requires_all_prerequisites(),
               "OpenCL residual route must require every prerequisite") ||
        !check(fused_residual_only_route_avoids_sparse_async_prerequisites(),
               "fused residual-only route must not require sparse async submission") ||
        !check(cache_storage_updates_avoid_rebuilding_disjoint_branches(),
               "disjoint branch caches must append without rebuilding old cache storage") ||
        !check(local_residual_graph_expands_pairs_in_attention_order(),
               "local residual pairs must expand mask-first without graph outputs") ||
        !check(no_allocation_tensor_avoids_sparse_destination_storage(),
               "residual pair allocation must back dense/cache roles but not metadata-only sparse") ||
        !check(no_allocation_scope_is_limited_to_local_sparse_output(),
               "no-allocation scope must detect dense/cache mistakes and reject invalid sparse ownership") ||
        !check(allocation_record_collector_preserves_pair_roles(),
               "pair allocation records must preserve role, graph, plan, and backing ownership") ||
        !check(local_residual_owned_sparse_requires_exact_pair_without_other_consumers(),
               "local metadata-only sparse outputs require an exact residual pair and no ordinary consumer") ||
        !check(local_lifetime_pair_state_and_pool_reuse_are_monotonic(),
               "local pair state and staging slots must complete monotonically and reuse bounded storage") ||
        !check(fused_pair_state_skips_sparse_event(),
               "fused pairs must enter residual ownership without a sparse event") ||
        !check(staging_pressure_waits_for_residual_publication_before_cache_wait(),
               "staging pressure must wait for residual publication before cache completion") ||
        !check(local_memory_plan_reports_allocator_ownership(),
               "local memory plan must report temporary and persistent ownership") ||
        !check(sd_htp_release_compute_buffer_after_residual_anchor(
                   SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, false),
               "synchronous residual anchor must release its compute arena") ||
        !check(!sd_htp_release_compute_buffer_after_residual_anchor(
                    SD_HTP_BLOCK_SPARSE_RESIDUAL_ANCHOR, true),
               "OpenCL residual anchor must preserve deferred arena ownership") ||
        !check(!sd_htp_release_compute_buffer_after_residual_anchor(
                    SD_HTP_BLOCK_SPARSE_RESIDUAL_APPLY, false),
               "sparse apply must keep the existing compute-buffer policy") ||
        !check(immediate.buffer_size * 2 < tail.buffer_size,
               "per-layer cache expansion must cut peak allocation by more than half")) {
        return 1;
    }
    std::puts("PASS graph-order");
    return 0;
}

int run_allocator_test() {
#if !defined(GGML_USE_HTP)
    std::puts("SKIP: allocator fixture requires an HTP-enabled build");
    return k_skip;
#else
    bool ok = true;
    for (int pair_count : {1, 2, 38}) {
        const cache_input_reserve_result result =
            build_cache_input_reserve_fixture(pair_count);
        std::printf(
            "allocator pairs=%d phase=%d runtime_bytes=%zu cache_bytes=%zu\n",
            pair_count, result.phase, result.runtime_bytes, result.cache_bytes);
        ok = check(result.executed,
                   "cache-input allocator fixture must execute") && ok;
        ok = check(result.runtime_bytes > 0,
                   "cache-input allocator fixture must reserve runtime storage") && ok;
        ok = check(result.runtime_bytes * 4 < result.cache_bytes,
                   "cache inputs must be bound before first reserve") && ok;
    }
    for (cache_input_failure_case failure_case : {
             cache_input_failure_case::first_reserve,
             cache_input_failure_case::second_alloc,
             cache_input_failure_case::preflight,
             cache_input_failure_case::compute,
         }) {
        const cache_input_reserve_result result =
            build_cache_input_reserve_fixture(1, failure_case);
        std::printf(
            "allocator failure=%d observed=%d restored=%d consumers=%d\n",
            static_cast<int>(failure_case),
            result.failure_observed ? 1 : 0,
            result.tensor_restored ? 1 : 0,
            result.consumers_released ? 1 : 0);
        ok = check(result.executed,
                   "cache-input failure fixture must execute") && ok;
        ok = check(result.failure_observed,
                   "cache-input failure fixture must reach its failure point") && ok;
        ok = check(result.tensor_restored,
                   "cache-input failure must restore tensor state") && ok;
        ok = check(result.consumers_released,
                   "cache-input failure must release async consumers") && ok;
    }
    if (!ok) {
        return 1;
    }
    std::puts("PASS allocator");
    return 0;
#endif
}

int run_packed_view_test() {
    const packed_view_reserve_result one =
        build_packed_view_reserve_fixture(1);
    const packed_view_reserve_result two =
        build_packed_view_reserve_fixture(2);
    const packed_view_reserve_result thirty_eight =
        build_packed_view_reserve_fixture(38);
    std::printf(
        "packed_view one_bytes=%zu two_bytes=%zu thirty_eight_bytes=%zu "
        "storage_bytes=%zu views=%d offsets=%d\n",
        one.buffer_bytes, two.buffer_bytes, thirty_eight.buffer_bytes,
        one.storage_bytes, thirty_eight.all_views ? 1 : 0,
        thirty_eight.offsets_ok ? 1 : 0);
    if (!check(one.reserved && two.reserved && thirty_eight.reserved,
               "packed-view allocator fixtures must reserve") ||
        !check(one.storage_count == 1 && two.storage_count == 2 &&
                   thirty_eight.storage_count == 38,
               "packed-view fixture must create every storage producer") ||
        !check(one.all_views && two.all_views && thirty_eight.all_views,
               "packed Q/K/V tensors must preserve GGML_OP_VIEW") ||
        !check(one.offsets_ok && two.offsets_ok && thirty_eight.offsets_ok,
               "packed Q/K/V offsets must remain unchanged") ||
        !check(thirty_eight.buffer_bytes < one.buffer_bytes * 3,
               "38-layer packed reserve must approach the one-layer peak")) {
        return 1;
    }
    std::puts("PASS packed-view");
    return 0;
}

#if defined(GGML_USE_HTP)

struct cache_pair {
    ggml_backend_htp_async_cache_slot * residual = nullptr;
    ggml_backend_htp_async_cache_slot * rho = nullptr;
};

void free_cache_pair(cache_pair & pair) {
    ggml_backend_htp_async_cache_slot_free(pair.rho);
    ggml_backend_htp_async_cache_slot_free(pair.residual);
    pair = {};
}

cache_pair create_cache_pair(ggml_backend_t backend, uint64_t generation) {
    constexpr int64_t residual_ne[GGML_MAX_DIMS] = { 16, 2, 4, 1 };
    constexpr int64_t rho_ne[GGML_MAX_DIMS] = { 1, 2, 4, 1 };
    cache_pair pair;
    char reason[512] = {};
    pair.residual = ggml_backend_htp_async_cache_slot_create(
        backend, "htp_block_sparse_residual_b0_self_s0", generation,
        GGML_TYPE_F16, residual_ne, reason, sizeof(reason));
    if (pair.residual == nullptr) {
        std::fprintf(stderr, "residual slot creation failed: %s\n", reason);
        return pair;
    }
    pair.rho = ggml_backend_htp_async_cache_slot_create(
        backend, "htp_block_sparse_residual_rho_b0_self_s0", generation,
        GGML_TYPE_F32, rho_ne, reason, sizeof(reason));
    if (pair.rho == nullptr) {
        std::fprintf(stderr, "rho slot creation failed: %s\n", reason);
        free_cache_pair(pair);
    }
    return pair;
}

int run_lifecycle_test() {
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_STRICT", "1", 1);
    if (!ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "SKIP: HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }

    ggml_backend_t backend = ggml_backend_htp_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: HTP backend initialization failed: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }

    cache_pair generation_1 = create_cache_pair(backend, 1);
    if (generation_1.residual == nullptr || generation_1.rho == nullptr) {
        ggml_backend_free(backend);
        return 1;
    }
    const size_t generation_1_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    std::printf("generation 1 imports=%zu\n", generation_1_imports);
    if (!check(generation_1_imports == 2, "generation 1 must own two imports")) {
        free_cache_pair(generation_1);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_htp_cache_slot_snapshot residual_snapshot = {};
    if (!check(ggml_backend_htp_async_cache_slot_snapshot(
                   generation_1.residual, &residual_snapshot),
               "async cache slot must expose a read-only snapshot") ||
        !check(std::strcmp(residual_snapshot.cache_key,
                          "htp_block_sparse_residual_b0_self_s0") == 0,
               "cache slot snapshot must preserve the key") ||
        !check(residual_snapshot.branch == 0 &&
                   residual_snapshot.generation == 1,
               "cache slot snapshot must preserve branch and generation") ||
        !check(residual_snapshot.fd >= 0 &&
                   residual_snapshot.logical_size > 0 &&
                   residual_snapshot.allocation_size >=
                       residual_snapshot.logical_size,
               "cache slot snapshot must expose exact RPCMEM allocation info") ||
        !check(residual_snapshot.opencl_import_identity != 0,
               "OpenCL cache slot snapshot must expose its import identity")) {
        free_cache_pair(generation_1);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_htp_lifetime_snapshot lifetime = {};
    if (!check(ggml_backend_htp_get_lifetime_snapshot(backend, &lifetime),
               "HTP backend must expose a lifetime snapshot") ||
        !check(lifetime.live_imports == 2 && lifetime.live_events == 0 &&
                   lifetime.live_leases == 0 && lifetime.pending_jobs == 0 &&
                   lifetime.retired_arenas == 0,
               "cache-only lifetime snapshot must report imports without graph work")) {
        free_cache_pair(generation_1);
        ggml_backend_free(backend);
        return 1;
    }

    char reason[512] = {};
    constexpr int64_t duplicate_ne[GGML_MAX_DIMS] = { 16, 2, 4, 1 };
    ggml_backend_htp_async_cache_slot * duplicate =
        ggml_backend_htp_async_cache_slot_create(
            backend, "htp_block_sparse_residual_b0_self_s0", 2,
            GGML_TYPE_F16, duplicate_ne, reason, sizeof(reason));
    if (!check(duplicate == nullptr,
               "a cache key must reject a second live generation")) {
        ggml_backend_htp_async_cache_slot_free(duplicate);
        free_cache_pair(generation_1);
        ggml_backend_free(backend);
        return 1;
    }

    free_cache_pair(generation_1);
    if (!check(ggml_backend_htp_opencl_live_cache_import_count(backend) == 0,
               "old generation imports must be released before replacement")) {
        ggml_backend_free(backend);
        return 1;
    }

    cache_pair generation_2 = create_cache_pair(backend, 2);
    const size_t generation_2_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    std::printf("generation 2 imports=%zu\n", generation_2_imports);
    if (generation_2.residual == nullptr || generation_2.rho == nullptr ||
        !check(generation_2_imports == 2,
               "generation 2 must replace generation 1 without import growth")) {
        free_cache_pair(generation_2);
        ggml_backend_free(backend);
        return 1;
    }

    free_cache_pair(generation_2);
    const size_t final_imports =
        ggml_backend_htp_opencl_live_cache_import_count(backend);
    const size_t final_events =
        ggml_backend_htp_opencl_live_cache_event_count(backend);
    std::printf("final imports=%zu events=%zu\n", final_imports, final_events);
    const bool clean =
        check(final_imports == 0, "all persistent imports must be released") &&
        check(final_events == 0, "all cache events must be released");
    lifetime = {};
    const bool lifetime_clean =
        check(ggml_backend_htp_get_lifetime_snapshot(backend, &lifetime),
              "final lifetime snapshot must be available") &&
        check(lifetime.live_imports == 0 && lifetime.live_events == 0 &&
                  lifetime.live_leases == 0 && lifetime.pending_jobs == 0 &&
                  lifetime.retired_arenas == 0,
              "final lifetime snapshot must be zero");
    ggml_backend_free(backend);
    if (!clean || !lifetime_clean) {
        return 1;
    }
    std::puts("PASS lifecycle");
    return 0;
}

void residual_only_noop(ggml_tensor *, int, int, void *) {}

int run_residual_only_test() {
    constexpr int64_t head_dim = 16;
    constexpr int64_t heads = 2;
    constexpr int64_t tokens = 4;
    constexpr int64_t rows = heads * tokens;
    constexpr int64_t packed_dim = head_dim + 1;
    constexpr size_t slab_elements = size_t(rows * packed_dim);
    constexpr uint64_t generation = 1;
    const char * residual_key = "htp_block_sparse_residual_b0_residual_only";
    const char * rho_key = "htp_block_sparse_residual_rho_b0_residual_only";

    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "0", 1);
    unsetenv("SD_HTP_SPARSE_OPENCL_ASYNC");
    unsetenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT");
    unsetenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD");

    if (!ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "SKIP: HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }
    ggml_backend_t backend = ggml_backend_htp_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: HTP backend initialization failed: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }

    ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_backend_buffer_t fused_buffer = nullptr;
    ggml_opencl_htp_sidecar * sidecar = nullptr;
    ggml_opencl_htp_graph * prepared = nullptr;
    ggml_backend_htp_async_cache_slot * residual_slot = nullptr;
    ggml_backend_htp_async_cache_slot * rho_slot = nullptr;
    bool residual_imported = false;
    bool rho_imported = false;
    int result = 1;
    char reason[1024] = {};

    do {
        if (!check(ctx != nullptr, "failed to create residual-only graph context")) break;
        ggml_tensor * fused = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, 2 * slab_elements);
        fused->op = GGML_OP_CUSTOM;
        ggml_set_name(fused, "htp_fused_anchor_attention");
        const size_t row_bytes = size_t(packed_dim) * sizeof(float);
        const size_t token_bytes = size_t(heads) * row_bytes;
        const size_t slab_bytes = size_t(tokens) * token_bytes;
        ggml_tensor * dense = ggml_view_4d(
            ctx, fused, packed_dim, heads, tokens, 1,
            row_bytes, token_bytes, slab_bytes, 0);
        ggml_tensor * sparse = ggml_view_4d(
            ctx, fused, packed_dim, heads, tokens, 1,
            row_bytes, token_bytes, slab_bytes, slab_bytes);
        ggml_set_name(dense, "htp_fused_anchor_dense_o_lse");
        ggml_set_name(sparse, "htp_fused_anchor_sparse_o_lse");

        fused_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(backend));
        if (!check(fused_buffer != nullptr, "failed to allocate fused RPCMEM backing")) break;

        std::vector<float> fused_data(2 * slab_elements);
        std::vector<float> rho_reference(rows);
        std::vector<float> residual_reference(size_t(rows * head_dim));
        for (int64_t row = 0; row < rows; ++row) {
            const float dense_lse = 1.0f + 0.03f * float(row);
            const float sparse_lse = dense_lse - 0.15f - 0.01f * float(row);
            const float rho = std::exp(sparse_lse - dense_lse);
            rho_reference[row] = rho;
            for (int64_t d = 0; d < head_dim; ++d) {
                const float dense_o = 0.25f + 0.01f * float(row) + 0.002f * float(d);
                const float sparse_o = -0.10f + 0.005f * float(row) - 0.001f * float(d);
                fused_data[size_t(row * packed_dim + d)] = dense_o;
                fused_data[slab_elements + size_t(row * packed_dim + d)] = sparse_o;
                residual_reference[size_t(row * head_dim + d)] = dense_o - rho * sparse_o;
            }
            fused_data[size_t(row * packed_dim + head_dim)] = dense_lse;
            fused_data[slab_elements + size_t(row * packed_dim + head_dim)] = sparse_lse;
        }
        ggml_backend_tensor_set(fused, fused_data.data(), 0,
                                fused_data.size() * sizeof(float));

        const int64_t residual_ne[GGML_MAX_DIMS] = {head_dim, heads, tokens, 1};
        const int64_t rho_ne[GGML_MAX_DIMS] = {1, heads, tokens, 1};
        residual_slot = ggml_backend_htp_async_cache_slot_create(
            backend, residual_key, generation, GGML_TYPE_F16,
            residual_ne, reason, sizeof(reason));
        rho_slot = ggml_backend_htp_async_cache_slot_create(
            backend, rho_key, generation, GGML_TYPE_F32,
            rho_ne, reason, sizeof(reason));
        if (!check(residual_slot != nullptr && rho_slot != nullptr,
                   "failed to create residual-only cache slots")) break;
        if (!check(
                ggml_backend_htp_async_cache_slot_opencl_import_identity(
                    residual_slot) != 0 &&
                ggml_backend_htp_async_cache_slot_opencl_import_identity(
                    rho_slot) != 0,
                "fused residual-only cache slots must own persistent imports")) {
            break;
        }
        ggml_tensor * residual =
            ggml_backend_htp_async_cache_slot_tensor(residual_slot);
        ggml_tensor * rho = ggml_backend_htp_async_cache_slot_tensor(rho_slot);
        residual->op = GGML_OP_CUSTOM;
        residual->src[0] = dense;
        residual->src[1] = sparse;
        residual->src[2] = rho;
        rho->op = GGML_OP_CUSTOM;
        rho->src[0] = dense;
        rho->src[1] = sparse;

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, rho);
        ggml_build_forward_expand(graph, residual);

        ggml_opencl_qcom_requirements requirements = {};
        sidecar = ggml_backend_opencl_htp_sidecar_create(
            &requirements, reason, sizeof(reason));
        if (sidecar == nullptr) {
            std::fprintf(stderr, "SKIP: residual-only sidecar unavailable: %s\n", reason);
            result = k_skip;
            break;
        }

        ggml_backend_htp_rpcmem_info fused_info = {};
        ggml_backend_htp_rpcmem_info residual_info = {};
        ggml_backend_htp_rpcmem_info rho_info = {};
        if (!check(ggml_backend_htp_get_rpcmem_info(fused->buffer, &fused_info) &&
                   ggml_backend_htp_async_cache_slot_rpcmem_info(residual_slot, &residual_info) &&
                   ggml_backend_htp_async_cache_slot_rpcmem_info(rho_slot, &rho_info),
                   "failed to query residual-only RPCMEM descriptors")) break;
        ggml_opencl_htp_import_descriptor descriptors[] = {
            {fused->buffer, fused_info.base, fused_info.fd,
             fused_info.logical_size, fused_info.allocation_size},
            {residual->buffer, residual_info.base, residual_info.fd,
             residual_info.logical_size, residual_info.allocation_size},
            {rho->buffer, rho_info.base, rho_info.fd,
             rho_info.logical_size, rho_info.allocation_size},
        };
        uint64_t import_identity = 0;
        if (!check(ggml_backend_opencl_htp_persistent_import_acquire(
                       sidecar, &descriptors[1], residual_key, generation,
                       &import_identity, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK,
                   "failed to import residual cache backing")) break;
        residual_imported = true;
        if (!check(ggml_backend_opencl_htp_persistent_import_acquire(
                       sidecar, &descriptors[2], rho_key, generation,
                       &import_identity, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK,
                   "failed to import rho cache backing")) break;
        rho_imported = true;

        if (!check(ggml_backend_opencl_htp_graph_preflight(
                       sidecar, graph, descriptors, 3, &prepared,
                       reason, sizeof(reason)) == GGML_OPENCL_HTP_OK,
                   reason)) break;
        if (!check(!ggml_backend_opencl_htp_graph_routes_node(prepared, fused) &&
                   ggml_backend_opencl_htp_graph_routes_node(prepared, rho) &&
                   ggml_backend_opencl_htp_graph_routes_node(prepared, residual) &&
                   ggml_backend_opencl_htp_graph_routed_node_count(prepared) == 2,
                   "residual-only preflight must route one pair and zero sparse nodes")) break;
        if (!check(ggml_backend_opencl_htp_graph_compute_exact_rho_residual_sync(
                       prepared, rho, reason, sizeof(reason)) == GGML_OPENCL_HTP_OK,
                   reason)) break;

        std::vector<float> rho_output(rows);
        std::vector<ggml_fp16_t> residual_output(size_t(rows * head_dim));
        ggml_backend_tensor_get(rho, rho_output.data(), 0,
                                rho_output.size() * sizeof(float));
        ggml_backend_tensor_get(residual, residual_output.data(), 0,
                                residual_output.size() * sizeof(ggml_fp16_t));
        float rho_max_abs = 0.0f;
        float residual_max_abs = 0.0f;
        double residual_mse = 0.0;
        bool finite = true;
        for (int64_t row = 0; row < rows; ++row) {
            finite = finite && std::isfinite(rho_output[row]);
            rho_max_abs = std::max(rho_max_abs,
                                   std::fabs(rho_output[row] - rho_reference[row]));
        }
        for (size_t i = 0; i < residual_output.size(); ++i) {
            const float value = ggml_fp16_to_fp32(residual_output[i]);
            const float error = value - residual_reference[i];
            finite = finite && std::isfinite(value);
            residual_max_abs = std::max(residual_max_abs, std::fabs(error));
            residual_mse += double(error) * error;
        }
        const float residual_rmse =
            std::sqrt(float(residual_mse / residual_output.size()));
        std::printf("residual-only sparse=0 residual=1 rho_max_abs=%.9g "
                    "residual_max_abs=%.9g residual_rmse=%.9g\n",
                    rho_max_abs, residual_max_abs, residual_rmse);
        if (!check(finite, "residual-only output contains NaN or Inf") ||
            !check(rho_max_abs <= 2e-3f, "residual-only rho exceeds tolerance") ||
            !check(residual_max_abs <= 5e-3f,
                   "residual-only FP16 max_abs exceeds tolerance") ||
            !check(residual_rmse <= 1e-3f,
                   "residual-only FP16 RMSE exceeds tolerance")) break;
        result = 0;
    } while (false);

    if (prepared != nullptr) {
        char finish_reason[512] = {};
        ggml_backend_opencl_htp_graph_finish(
            prepared, finish_reason, sizeof(finish_reason));
        ggml_backend_opencl_htp_graph_free(&prepared);
    }
    if (rho_imported) {
        ggml_backend_opencl_htp_persistent_import_release(
            sidecar, rho_key, generation, reason, sizeof(reason));
    }
    if (residual_imported) {
        ggml_backend_opencl_htp_persistent_import_release(
            sidecar, residual_key, generation, reason, sizeof(reason));
    }
    if (sidecar != nullptr) {
        const size_t imports =
            ggml_backend_opencl_htp_sidecar_live_persistent_import_count(sidecar);
        const size_t events =
            ggml_backend_opencl_htp_sidecar_live_cache_event_count(sidecar);
        if (result == 0 && (!check(imports == 0, "residual-only imports leaked") ||
                            !check(events == 0, "residual-only events leaked"))) {
            result = 1;
        }
        ggml_backend_opencl_htp_sidecar_free(sidecar);
    }
    ggml_backend_htp_async_cache_slot_free(rho_slot);
    ggml_backend_htp_async_cache_slot_free(residual_slot);
    if (fused_buffer != nullptr) ggml_backend_buffer_free(fused_buffer);
    if (ctx != nullptr) ggml_free(ctx);
    ggml_backend_free(backend);
    if (result == 0) std::puts("PASS residual-only");
    return result;
}

enum class sparse_preflight_case {
    metadata_only_sparse,
    dense_no_alloc,
    ordinary_sparse_consumer,
    deferred_stage_noop,
};

int run_sparse_residual_preflight_case(sparse_preflight_case test_case) {
    constexpr int64_t head_dim = 64;
    constexpr int64_t heads = 1;
    constexpr int64_t tokens = 32;
    constexpr uint64_t generation = 1;
    const char * residual_key =
        "htp_block_sparse_residual_b0_preflight_s0";
    const char * rho_key =
        "htp_block_sparse_residual_rho_b0_preflight_s0";
    const bool deferred_stage_noop =
        test_case == sparse_preflight_case::deferred_stage_noop;
    setenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE",
           deferred_stage_noop ? "deferred" : "reuse", 1);

    ggml_backend_t backend = ggml_backend_htp_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: HTP backend initialization failed: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }

    ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_gallocr_t allocator = nullptr;
    ggml_opencl_htp_sidecar * sidecar = nullptr;
    ggml_opencl_htp_graph * prepared = nullptr;
    ggml_backend_htp_async_cache_slot * residual_slot = nullptr;
    ggml_backend_htp_async_cache_slot * rho_slot = nullptr;
    bool residual_imported = false;
    bool rho_imported = false;
    int result = 1;
    char reason[1024] = {};

    do {
        if (!check(ctx != nullptr, "failed to create sparse preflight context")) break;

        ggml_tensor * q = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, head_dim, tokens, heads, 1);
        ggml_tensor * k = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, head_dim, tokens, heads, 1);
        ggml_tensor * v = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, head_dim, tokens, heads, 1);
        ggml_tensor * mask = ggml_new_tensor_4d(
            ctx, GGML_TYPE_I32, 1, heads, 1, 1);
        ggml_set_name(q, "preflight_q");
        ggml_set_name(k, "preflight_k");
        ggml_set_name(v, "preflight_v");
        ggml_set_name(mask, "preflight_block_index");

        ggml_tensor * sparse = ggml_flash_attn_ext_with_lse(
            ctx, q, k, v, mask, 1.0f / std::sqrt(float(head_dim)),
            0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(sparse, GGML_PREC_F32);
        ggml_flash_attn_ext_set_block_sparse(
            sparse, 32, 32, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        ggml_set_name(sparse, "preflight_sparse_o_lse");

        ggml_tensor * dense = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, head_dim + 1, heads, tokens, 1);
        dense->op = GGML_OP_CUSTOM;
        dense->src[0] = q;
        dense->src[1] = k;
        dense->src[2] = v;
        ggml_set_name(dense, "preflight_dense_o_lse");

        const int64_t residual_ne[GGML_MAX_DIMS] = {
            head_dim, heads, tokens, 1,
        };
        const int64_t rho_ne[GGML_MAX_DIMS] = {1, heads, tokens, 1};
        residual_slot = ggml_backend_htp_async_cache_slot_create(
            backend, residual_key, generation, GGML_TYPE_F16,
            residual_ne, reason, sizeof(reason));
        rho_slot = ggml_backend_htp_async_cache_slot_create(
            backend, rho_key, generation, GGML_TYPE_F32,
            rho_ne, reason, sizeof(reason));
        if (!check(residual_slot != nullptr && rho_slot != nullptr,
                   "failed to create sparse preflight cache slots")) break;

        ggml_tensor * residual =
            ggml_backend_htp_async_cache_slot_tensor(residual_slot);
        ggml_tensor * rho =
            ggml_backend_htp_async_cache_slot_tensor(rho_slot);
        residual->op = GGML_OP_CUSTOM;
        residual->src[0] = dense;
        residual->src[1] = sparse;
        residual->src[2] = rho;
        rho->op = GGML_OP_CUSTOM;
        rho->src[0] = dense;
        rho->src[1] = sparse;

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, rho);
        ggml_build_forward_expand(graph, residual);
        if (test_case == sparse_preflight_case::ordinary_sparse_consumer) {
            ggml_tensor * consumer = ggml_new_tensor_1d(
                ctx, GGML_TYPE_F32, 1);
            consumer->op = GGML_OP_CUSTOM;
            consumer->src[0] = sparse;
            ggml_set_name(consumer, "preflight_sparse_ordinary_consumer");
            ggml_build_forward_expand(graph, consumer);
        }

        allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        const ggml_tensor * no_alloc[] = {sparse, dense};
        const size_t no_alloc_count = deferred_stage_noop ? 0 :
            (test_case == sparse_preflight_case::dense_no_alloc ? 2 : 1);
        if (!check(allocator != nullptr &&
                   ggml_gallocr_set_no_alloc_tensors(
                       allocator, no_alloc_count > 0 ? no_alloc : nullptr,
                       no_alloc_count) &&
                   ggml_gallocr_reserve(allocator, graph) &&
                   ggml_gallocr_alloc_graph(allocator, graph),
                   "failed to allocate sparse preflight graph")) break;
        if (deferred_stage_noop) {
            if (!check(tensor_has_resolved_backing(sparse),
                       "deferred sparse output must have backing")) break;
        } else if (!check(sparse->buffer == nullptr && sparse->data == nullptr,
                          "sparse preflight output must remain metadata-only")) {
            break;
        }
        if (test_case != sparse_preflight_case::dense_no_alloc &&
            !check(tensor_has_resolved_backing(dense),
                   "sparse preflight dense output must have backing")) break;

        ggml_opencl_qcom_requirements requirements = {};
        sidecar = ggml_backend_opencl_htp_sidecar_create(
            &requirements, reason, sizeof(reason));
        if (sidecar == nullptr) {
            std::fprintf(stderr, "SKIP: sparse preflight sidecar unavailable: %s\n",
                         reason);
            result = k_skip;
            break;
        }

        ggml_backend_htp_rpcmem_info compute_info = {};
        ggml_backend_htp_rpcmem_info residual_info = {};
        ggml_backend_htp_rpcmem_info rho_info = {};
        const bool compute_rpcmem = ggml_backend_htp_get_rpcmem_info(
            q->buffer, &compute_info);
        const bool residual_rpcmem =
            ggml_backend_htp_async_cache_slot_rpcmem_info(
                residual_slot, &residual_info);
        const bool rho_rpcmem = ggml_backend_htp_async_cache_slot_rpcmem_info(
            rho_slot, &rho_info);
        if (!compute_rpcmem || !residual_rpcmem || !rho_rpcmem) {
            std::fprintf(
                stderr,
                "FAIL: sparse preflight RPCMEM descriptors "
                "compute=%d residual=%d rho=%d dense_buffer=%p\n",
                compute_rpcmem, residual_rpcmem, rho_rpcmem,
                static_cast<void *>(q->buffer));
            break;
        }
        ggml_opencl_htp_import_descriptor descriptors[] = {
            {q->buffer, compute_info.base, compute_info.fd,
             compute_info.logical_size, compute_info.allocation_size},
            {residual->buffer, residual_info.base, residual_info.fd,
             residual_info.logical_size, residual_info.allocation_size},
            {rho->buffer, rho_info.base, rho_info.fd,
             rho_info.logical_size, rho_info.allocation_size},
        };
        uint64_t import_identity = 0;
        if (!check(ggml_backend_opencl_htp_persistent_import_acquire(
                       sidecar, &descriptors[1], residual_key, generation,
                       &import_identity, reason, sizeof(reason)) ==
                       GGML_OPENCL_HTP_OK,
                   "failed to import sparse preflight residual backing")) break;
        residual_imported = true;
        if (!check(ggml_backend_opencl_htp_persistent_import_acquire(
                       sidecar, &descriptors[2], rho_key, generation,
                       &import_identity, reason, sizeof(reason)) ==
                       GGML_OPENCL_HTP_OK,
                   "failed to import sparse preflight rho backing")) break;
        rho_imported = true;

        const ggml_opencl_htp_result preflight =
            ggml_backend_opencl_htp_graph_preflight(
                sidecar, graph, descriptors, 3, &prepared,
                reason, sizeof(reason));
        if (test_case == sparse_preflight_case::metadata_only_sparse) {
            if (!check(preflight == GGML_OPENCL_HTP_OK, reason) ||
                !check(ggml_backend_opencl_htp_graph_sparse_output_is_residual_owned(
                           prepared, sparse),
                       "metadata-only sparse output must be residual-owned")) {
                break;
            }
        } else if (deferred_stage_noop) {
            if (!check(preflight == GGML_OPENCL_HTP_OK, reason) ||
                !check(!ggml_backend_opencl_htp_graph_sparse_output_is_residual_owned(
                            prepared, sparse),
                       "deferred sparse output must keep graph-owned storage") ||
                !check(ggml_backend_opencl_htp_graph_stage_block_sparse_attn_mask(
                            prepared, sparse, reason, sizeof(reason)) ==
                            GGML_OPENCL_HTP_OK,
                       reason)) {
                break;
            }
        } else {
            const char * expected_role =
                test_case == sparse_preflight_case::dense_no_alloc
                    ? "dense" : "sparse";
            if (!check(preflight == GGML_OPENCL_HTP_UNSUPPORTED,
                       "invalid sparse preflight case must fail closed") ||
                !check(std::strstr(reason, expected_role) != nullptr &&
                           std::strstr(reason, "no backing buffer") != nullptr,
                       reason)) {
                break;
            }
        }
        result = 0;
    } while (false);

    if (prepared != nullptr) {
        char finish_reason[512] = {};
        if (ggml_backend_opencl_htp_graph_finish(
                prepared, finish_reason, sizeof(finish_reason)) !=
                GGML_OPENCL_HTP_OK && result == 0) {
            std::fprintf(stderr, "FAIL: sparse preflight finish: %s\n",
                         finish_reason);
            result = 1;
        }
        ggml_backend_opencl_htp_graph_free(&prepared);
    }
    if (rho_imported) {
        ggml_backend_opencl_htp_persistent_import_release(
            sidecar, rho_key, generation, reason, sizeof(reason));
    }
    if (residual_imported) {
        ggml_backend_opencl_htp_persistent_import_release(
            sidecar, residual_key, generation, reason, sizeof(reason));
    }
    if (sidecar != nullptr) {
        const size_t imports =
            ggml_backend_opencl_htp_sidecar_live_persistent_import_count(sidecar);
        const size_t events =
            ggml_backend_opencl_htp_sidecar_live_cache_event_count(sidecar);
        if (result == 0 &&
            (!check(imports == 0, "sparse preflight imports leaked") ||
             !check(events == 0, "sparse preflight events leaked"))) {
            result = 1;
        }
        ggml_backend_opencl_htp_sidecar_free(sidecar);
    }
    ggml_backend_htp_async_cache_slot_free(rho_slot);
    ggml_backend_htp_async_cache_slot_free(residual_slot);
    if (allocator != nullptr) ggml_gallocr_free(allocator);
    if (ctx != nullptr) ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

int run_sparse_residual_preflight_test() {
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_CACHE", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_RESIDUAL_EXACT_RHO", "1", 1);
    setenv("SD_HTP_BLOCK_SPARSE_FUSED_ANCHOR", "0", 1);
    setenv("SD_HTP_SPARSE_OPENCL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_RESIDUAL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_ASYNC", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_EARLY_SUBMIT", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_SUBMIT_THREAD", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_DEFER_RESIDUAL", "1", 1);
    setenv("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE", "reuse", 1);

    if (!ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "SKIP: HTP runtime unavailable: %s\n",
                     ggml_backend_htp_runtime_error());
        return k_skip;
    }
    for (sparse_preflight_case test_case : {
             sparse_preflight_case::metadata_only_sparse,
             sparse_preflight_case::dense_no_alloc,
             sparse_preflight_case::ordinary_sparse_consumer,
             sparse_preflight_case::deferred_stage_noop,
         }) {
        const int case_result = run_sparse_residual_preflight_case(test_case);
        if (case_result != 0) {
            return case_result;
        }
    }
    std::puts("PASS sparse-residual-preflight");
    return 0;
}

#else

int run_lifecycle_test() {
    std::puts("SKIP: test was built without HTP support");
    return k_skip;
}

int run_residual_only_test() {
    std::puts("SKIP: test was built without HTP support");
    return k_skip;
}

int run_sparse_residual_preflight_test() {
    std::puts("SKIP: test was built without HTP support");
    return k_skip;
}

#endif

}  // namespace

int main() {
    const char * mode = std::getenv("HTP_RESIDUAL_CACHE_TEST");
    if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "fp16") == 0) {
        return run_fp16_test();
    }
    if (std::strcmp(mode, "lifecycle") == 0) {
        return run_lifecycle_test();
    }
    if (std::strcmp(mode, "graph-order") == 0) {
        return run_graph_order_test();
    }
    if (std::strcmp(mode, "allocator") == 0) {
        return run_allocator_test();
    }
    if (std::strcmp(mode, "packed-view") == 0) {
        return run_packed_view_test();
    }
    if (std::strcmp(mode, "residual-only") == 0) {
        return run_residual_only_test();
    }
    if (std::strcmp(mode, "sparse-residual-preflight") == 0) {
        return run_sparse_residual_preflight_test();
    }
    std::fprintf(stderr, "unknown HTP_RESIDUAL_CACHE_TEST mode: %s\n", mode);
    return 2;
}
