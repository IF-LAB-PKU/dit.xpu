#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-htp.h>

#include <cstdint>
#include <vector>

namespace htp_attention_common {

constexpr int kInputDim = 128;
constexpr int kImageTokens = 256;
constexpr int kTextTokens = 32;
constexpr int kBlock = 32;

struct ImageSize {
    int resolution;
    int image_tokens;
    int block_rows;
    int block_cols;
};

struct CaseSpec {
    const char * model;
    const char * family;
    int heads;
    int head_dim;
    int image_tokens;
    int text_tokens;
    bool dual;
    bool bias;
    bool rope;
    bool full_keep;
    bool sparse;
    int resolution = 256;
    int image_block_rows = 4;
    int image_block_cols = 2;
    bool dense_inference = false;
    bool external_inference = false;
    bool packed_inference = false;
    bool head_major_v_input = false;
    bool head_major_qkv_input = false;
    bool residual_apply = false;
};

enum class Route {
    Dense,
    ExternalSparseL2,
    ExternalSparseBaseline,
    ExternalSparseChunk,
    PackedSparse,
    FusedAnchor,
};

struct BranchData {
    int tokens = 0;
    std::vector<float> activation;
    std::vector<ggml_fp16_t> weight;
    std::vector<ggml_fp16_t> packed_weight;
    std::vector<float> bias;
    std::vector<float> q_gamma;
    std::vector<float> k_gamma;
};

struct Reference {
    std::vector<float> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
};

struct AttentionReference {
    std::vector<float> output;
    std::vector<float> lse;
};

struct InputData {
    std::vector<BranchData> branches;
    std::vector<float> rope;
    std::vector<int32_t> block_index;
    std::vector<float> residual;
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * debug_q = nullptr;
    ggml_tensor * debug_k = nullptr;
    ggml_tensor * debug_v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * packed_storage = nullptr;
    ggml_tensor * fused_storage = nullptr;
    ggml_tensor * dense_packed = nullptr;
    ggml_tensor * sparse_packed = nullptr;
    ggml_tensor * dense_output = nullptr;
    ggml_tensor * dense_lse = nullptr;
    ggml_tensor * sparse_output = nullptr;
    ggml_tensor * sparse_lse = nullptr;
    ggml_tensor * rho = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * apply_residual = nullptr;
    ggml_tensor * rope = nullptr;
    ggml_tensor * built_in_one = nullptr;
    std::vector<ggml_tensor *> activations;
    std::vector<ggml_tensor *> weights;
    std::vector<ggml_tensor *> biases;
    std::vector<ggml_tensor *> q_gammas;
    std::vector<ggml_tensor *> k_gammas;
    Route route = Route::Dense;
};

const ImageSize * image_sizes();
int image_size_count();
const CaseSpec * cases();
int case_count();
int total_tokens(const CaseSpec & spec);
bool selected(const ImageSize & size, const CaseSpec & spec);
CaseSpec make_inference_case(const ImageSize & size, const CaseSpec & base, Route route);

InputData make_input(const CaseSpec & spec);
void append_reference_branch(const CaseSpec & spec, const BranchData & branch,
                             bool sparse, Reference & ref);
std::vector<int32_t> make_block_index(const CaseSpec & spec);
std::vector<float> reference_attention(const CaseSpec & spec, const Reference & ref,
                                       const std::vector<int32_t> * index);
AttentionReference reference_attention_with_lse(
    const CaseSpec & spec, const Reference & ref,
    const std::vector<int32_t> * index);

Graph build_graph(const CaseSpec & spec, Route route, ggml_backend_t backend = nullptr);
bool upload_graph(Graph & graph, const InputData & data, ggml_backend_t backend);
bool graph_contract(const Graph & graph, const CaseSpec & spec);
void free_graph(Graph & graph);

// Compare the realistic packed and reordered Z-Image attention subgraphs.
bool run_z_image_route_pair(ggml_backend_t htp,
                            const ImageSize & size,
                            const CaseSpec & base);

// Host-only RED/GREEN contract test used during fixture migration.
int run_fixture_contract_test();
int run_hybrid_placement_contract_test(ggml_backend_t htp);

} // namespace htp_attention_common
