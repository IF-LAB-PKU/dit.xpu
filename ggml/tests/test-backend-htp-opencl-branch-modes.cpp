// Single-attention branch strategy matrix for the HTP dense + OpenCL sparse pair.
//
// The test drives the OpenCL sparse sidecar through its public API only and it
// enables no production route: the one production hook it reads is the HTP
// attention profile (device elapsed time plus the DSP doorbell instant), which
// exists for exactly this kind of measurement.  Each
// line runs exactly one NPU dense attention and one block-sparse attention -
// through the sidecar on every line but npu-serial, where both nodes are HTP
// nodes - and then finishes the residual on the main path, so every number
// below belongs to a single attention pair instead of a denoise loop.
//
// Both branches run the routes the kernel benchmarks measure: the dense
// attention (f16 mask, every real key visible, no block index) is the NPU node,
// and the block-sparse attention - the production one on the OpenCL lines, the
// HTP kernel on npu-serial - carries a block index that keeps a 3x3
// neighbourhood of image blocks plus the protected text blocks, so it really is
// cheaper than the dense one and the residual bridges the two.
// HTP_OPENCL_BRANCH_MASK=all switches the block index to "keep everything",
// which makes both branches the same full attention.
//
//   npu-serial       Route 1 of the timeline figure: the NPU dense node and the
//                    NPU sparse node run one after the other on the HTP
//                    backend (both nodes are block-sparse-capable HTP attention
//                    kernels, the sparse one carries the block index), and the
//                    residual is finished on the CPU.  No OpenCL work at all.
//   parallel-copy    Route 2: two rpcmem allocations and the explicit copy
//                    in/out.  The main thread submits the GPU sparse work
//                    before the NPU starts the dense node: one plain enqueue on
//                    the calling thread.  No submission queue, no submission
//                    thread, no early-submit bookkeeping; the wait stays
//                    deferred until after the NPU dense node.  Residual on CPU.
//   parallel-shared  Route 3, the production fast path: one shared rpcmem
//                    allocation,
//                    zero-copy imports in both directions, early submission
//                    from the sidecar submission thread, deferred wait, and the
//                    residual published by that same submission thread as the
//                    production tagged residual pair (rho and residual rpcmem
//                    cache slots, exact-rho kernel) once the NPU dense node has
//                    finished.
//
// What the test reports:
//
// 1. The stage timeline, i.e. the per-stage numbers the figure draws.  Every
//    line reports its stages as labelled spans of the pair window, one
//    `stage line=...` row per drawn box, the per-kind sums on `stage_totals`,
//    and one `timeline line=...` ASCII bar per lane:
//
//   name="dense compute"     the figure box the span stands for
//   lane=main|gpu            the figure's row: the main path or the GPU branch
//   kind=host|npu|gpu|waiting|others
//                            the figure's legend entry: host submit/sync/notify
//                            (orange), NPU compute (green), GPU compute (blue),
//                            waiting (hatched), the rest (grey)
//   begin_ms / end_ms        position on the pair timeline, in ms from the start
//                            of the pair (`begin_ms=0`)
//   dur_ms / share           duration and its fraction of the pair
//
// 2. The explicit copies, as one `copy line=...` row per line:
//
//   copy_in_mb / copy_in_ms    payload moved into the second rpcmem allocation
//                              and the time it took
//   copy_out_mb / copy_out_ms  result moved back and the time it took
//   copy_mb / copy_ms          both together
//
// Nothing else is reported: no per-column dispatch table, no derived main-path
// decomposition.  The staged sums the test computes along the way (the NPU
// device time behind a submit/compute/sync split, the CL kernel time, the
// residual windows) are inputs to the boxes above and are not printed on their
// own.
//
// The spans of one line tile the pair window and never overlap, so the figure
// can be drawn straight from them.  What each name means on which line:
//
//   submit dense             host work in front of the NPU dense dispatch: the
//                            per-node argument preparation, the request memcpy
//                            and checksum into the shared message channel, and
//                            the doorbell store that hands the node to the DSP
//   dense compute            NPU dense attention: the device time the DSP
//                            itself reports for the node
//   sync dense               host tail of the dispatch window, what is left of
//                            it once the submit and the reported device time
//                            are taken out: the DSP setup and the signal
//                            latency the node itself does not account for
//   submit sparse            the same three boxes for the route-1 NPU sparse
//   sync sparse              node (npu-serial line only)
//   sparse compute           the sparse attention itself: NPU on route 1 (the
//                            device time of that node), OpenCL kernel time from
//                            the CL event timestamps on routes 2 and 3
//   submit sparse            the main path handing the GPU sparse work over
//                            (enqueue + flush, from the CL profile)
//   sync dense+sparse        the main path blocking on the GPU sparse producer
//                            after the dense node (CL event wait)
//   wait dense               route 3 GPU branch idle between the end of the
//                            sparse kernel and the end of the dense node
//   submit residual          main path queueing the residual job
//   residual publish         submission thread publishing the queued job
//   residual compute         GPU branch running the exact-rho residual
//   cpu residual compute     host residual: readback, rho, residual, writeback
//   copy in / copy out       explicit rpcmem <-> rpcmem transfers on the lines
//                            without shared memory
//   others                   the remainder of the pair window: in production
//                            the computation between two attentions, here the
//                            host bookkeeping between the measured stages
//
// Every span is on exactly one lane; the two lanes are reported separately so
// both rows of the figure can be drawn from the same line of output.  Spans are
// printed for every raw run and for the mean line.  HTP_OPENCL_BRANCH_TIMELINE=0
// suppresses the `timeline` bars.
//
// The payload defaults to the SD3.5 Medium joint (image <-> text) attention at
// 1024x1024: 16x16 latent patches give 4096 image tokens, the text context adds
// 154 tokens, and the MMdiT block uses 24 heads of head_dim 64.  The attention
// runs over the real 4096 + 154 = 4250 joint sequence (133 sparse blocks, the
// last one partial); only the QKV storage is padded to 4256 so the sparse core
// can read its final block.
//
// Usage:
//   HTP_OPENCL_BRANCH_MODES=all|npu-serial|parallel-copy|parallel-shared
//   HTP_OPENCL_BRANCH_IMAGE_TOKENS=4096  (multiple of the 32x32 sparse block)
//   HTP_OPENCL_BRANCH_TEXT_TOKENS=154    (protected text context)
//   HTP_OPENCL_BRANCH_IMAGE_BLOCK_ROWS=16
//   HTP_OPENCL_BRANCH_IMAGE_BLOCK_COLS=8
//   HTP_OPENCL_BRANCH_HEADS=24
//   HTP_OPENCL_BRANCH_KV_HEADS=24
//   HTP_OPENCL_BRANCH_HEAD_DIM=64         (64 or 128)
//   HTP_OPENCL_BRANCH_REPEAT=3
//   HTP_OPENCL_BRANCH_TRACE=1             sidecar trace on stderr
//   HTP_OPENCL_BRANCH_MASK=halo|all       sparse block index (default halo)
//   HTP_OPENCL_BRANCH_TIMELINE=1          ASCII stage bars (default on)
// Every raw run is printed and the summary reports the mean of each stage,
// because the host-side stages swing with scheduling.
// Exit codes: 0 pass, 1 comparison failure, 2 setup failure, 77 unsupported.

#include "ggml-backend.h"
#include "ggml-htp.h"
#include "ggml-opencl.h"
#include "ggml-opencl-htp.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int k_block = 32;  // the sparse core only accepts 32x32 blocks
constexpr int k_skip = 77;
constexpr float k_tolerance = 1e-2f;  // the packed lse column carries ~5e-3 device precision
constexpr float k_rho_tolerance = 1.5e-2f;  // both branches carry ~5e-3 lse precision
constexpr float k_residual_tolerance = 3e-2f;  // OpenCL residual vs the host reference
constexpr size_t k_reason_capacity = 1024;

int env_int(const char * name, int fallback, int minimum) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0' || parsed < minimum) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

void set_env(const char * name, const char * value) {
    setenv(name, value, 1);
}

void unset_env(const char * name) {
    unsetenv(name);
}

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool check_result(ggml_opencl_htp_result result, const char * reason, const char * what) {
    if (result != GGML_OPENCL_HTP_OK) {
        std::fprintf(stderr, "FAIL: %s (result=%d reason=%s)\n", what, static_cast<int>(result),
                     reason[0] != '\0' ? reason : "unknown");
        return false;
    }
    return true;
}

int64_t now_us() {
    return ggml_time_us();
}

// ---------------------------------------------------------------------------
// Attention case + CPU reference
// ---------------------------------------------------------------------------

// Which attention the two branches compute.  `halo` is the production route the
// kernel benchmarks use: the sparse branch only sees a 3x3 neighbourhood of
// image blocks plus the protected text blocks, so it is a genuinely cheaper
// attention than the dense branch and the residual is what bridges the two.
// `all` keeps every block, which makes both branches compute the same full
// attention (the mode the earlier numbers were taken in).
enum class mask_mode { halo, all_kept };

mask_mode mask_mode_from_env() {
    const char * value = std::getenv("HTP_OPENCL_BRANCH_MASK");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "halo") == 0) {
        return mask_mode::halo;
    }
    if (std::strcmp(value, "all") == 0) {
        return mask_mode::all_kept;
    }
    std::fprintf(stderr, "FAIL: unknown HTP_OPENCL_BRANCH_MASK=%s (halo|all)\n", value);
    return mask_mode::halo;
}

struct attention_case {
    int head_dim = 64;
    int heads = 24;
    int kv_heads = 24;
    int batch = 1;
    // SD3.5 Medium @1024: 4096 image tokens, then the 154-token text context
    // the conditioner produces (clip_l 77 + clip_g 77 padded to 4096 columns,
    // concatenated with t5's 77 tokens along the sequence).  The sparse core
    // works on 32x32 blocks, so the sequence is block padded to 4096 + 160.
    int image_tokens = 4096;
    int text_tokens = 154;   // real context length
    int tokens = 4096 + 154; // joint sequence the attention runs over
    // The sparse core only reads whole 32x32 blocks, so the QKV storage is
    // padded up to the next block boundary while the attention keeps the real
    // length.  ggml_flash_attn_ext_set_static_halo_2d asserts
    // src[0]->ne[1] == image_tokens + text_tokens, so the padding may never
    // reach the attention node itself.
    int tokens_alloc = 4096 + 160;
    // 1024x1024 -> 64x64 patches -> 16x8 grid of 32x32 sparse blocks.
    int image_block_rows = 16;
    int image_block_cols = 8;
    std::vector<float> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<int32_t> mask;
    // Packed rows, [head_dim + 1] each: the full attention the dense branch
    // computes and the masked attention the sparse branch computes.  Under the
    // all-kept mask the two are the same computation.
    std::vector<float> reference;
    std::vector<float> sparse_reference;
    mask_mode mode = mask_mode::halo;
};

int q_blocks(const attention_case & test) {
    return (test.tokens + k_block - 1) / k_block;
}

int image_blocks(const attention_case & test) {
    return test.image_tokens / k_block;
}

int real_tokens(const attention_case & test) {
    return test.image_tokens + test.text_tokens;
}

// Storage carries `tokens_alloc` tokens so the sparse core can read the block
// that covers the tail of the joint sequence; the attention branches are fed by
// a truncated view of that storage.
ggml_tensor * logical_tokens_view(ggml_context * ctx, ggml_tensor * storage,
                                  const attention_case & test) {
    return ggml_view_4d(ctx, storage, storage->ne[0], storage->ne[1], test.tokens,
                        storage->ne[3], storage->nb[1], storage->nb[2], storage->nb[3], 0);
}

size_t packed_rows(const attention_case & test) {
    return static_cast<size_t>(test.batch) * test.tokens * test.heads;
}

size_t packed_index(const attention_case & test, int b, int token, int head, int d) {
    return (static_cast<size_t>((b * test.tokens + token) * test.heads + head) *
            (test.head_dim + 1)) + d;
}

// Storage tensors are [D, heads, tokens, batch] and the attention views are
// [D, tokens, heads, batch].  ggml_backend_tensor_set copies linearly, so these
// offsets are the linear order of the [D, heads, tokens] storage, which equals
// the token-major order the reference math and the OpenCL kernel index with.
size_t q_index(const attention_case & test, int b, int token, int head, int d) {
    return ((static_cast<size_t>(b * test.tokens + token) * test.heads + head) *
            test.head_dim) + d;
}

size_t kv_index(const attention_case & test, int b, int token, int head, int d) {
    return ((static_cast<size_t>(b * test.tokens + token) * test.kv_heads + head) *
            test.head_dim) + d;
}

// Production block index, same shape the kernel benchmarks build: image query
// blocks keep a 3x3 neighbourhood of image blocks, text query blocks keep
// everything, and every row is terminated with -1.  `all_kept` keeps every
// block for every query instead, which turns the sparse branch into the dense
// attention.
void fill_block_index(attention_case & test) {
    const int qb_count = q_blocks(test);
    const int img_blocks = image_blocks(test);
    const int columns = qb_count + 1;
    test.mask.assign(static_cast<size_t>(test.heads) * qb_count * columns, -1);
    for (int head = 0; head < test.heads; ++head) {
        const int kv_head = head / (test.heads / test.kv_heads);
        for (int qb = 0; qb < qb_count; ++qb) {
            int32_t * row = test.mask.data() +
                (static_cast<size_t>(head * qb_count + qb) * columns);
            int kept = 0;
            if (test.mode == mask_mode::all_kept || qb >= img_blocks) {
                for (int kb = 0; kb < qb_count; ++kb) {
                    row[kept++] = kv_head * qb_count + kb;
                }
                continue;
            }
            const int qy = qb / test.image_block_cols;
            const int qx = qb % test.image_block_cols;
            for (int ky = std::max(0, qy - 1); ky <= std::min(test.image_block_rows - 1, qy + 1); ++ky) {
                for (int kx = std::max(0, qx - 1); kx <= std::min(test.image_block_cols - 1, qx + 1); ++kx) {
                    row[kept++] = kv_head * qb_count + ky * test.image_block_cols + kx;
                }
            }
            for (int kb = img_blocks; kb < qb_count; ++kb) {
                row[kept++] = kv_head * qb_count + kb;
            }
        }
    }
}

// Masked attention reference: `index == nullptr` is the dense branch (every
// key token), otherwise only the tokens of the blocks the row keeps, which is
// how the sparse kernels apply the index.
void compute_reference(
        const attention_case & test,
        const std::vector<int32_t> * index,
        std::vector<float> * out) {
    const int qb_count = q_blocks(test);
    const int columns = qb_count + 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(test.head_dim));
    out->assign(packed_rows(test) * static_cast<size_t>(test.head_dim + 1), 0.0f);
    std::vector<int> keys;
    keys.reserve(static_cast<size_t>(test.tokens));
    std::vector<double> score(static_cast<size_t>(test.tokens), 0.0);
    for (int b = 0; b < test.batch; ++b) {
        for (int token = 0; token < test.tokens; ++token) {
            for (int head = 0; head < test.heads; ++head) {
                const int kv_head = head / (test.heads / test.kv_heads);
                keys.clear();
                if (index == nullptr) {
                    for (int key = 0; key < test.tokens; ++key) {
                        keys.push_back(key);
                    }
                } else {
                    const int32_t * row = index->data() +
                        (static_cast<size_t>(head * qb_count + token / k_block) * columns);
                    for (int p = 0; p < columns && row[p] >= 0; ++p) {
                        const int begin = (row[p] % qb_count) * k_block;
                        const int end = std::min(test.tokens, begin + k_block);
                        for (int key = begin; key < end; ++key) {
                            keys.push_back(key);
                        }
                    }
                }
                double max_score = -1e30;
                for (size_t i = 0; i < keys.size(); ++i) {
                    const int key = keys[i];
                    double dot = 0.0;
                    for (int d = 0; d < test.head_dim; ++d) {
                        dot += static_cast<double>(test.q[q_index(test, b, token, head, d)]) *
                            ggml_fp16_to_fp32(test.k[kv_index(test, b, key, kv_head, d)]);
                    }
                    score[i] = dot * scale;
                    max_score = std::max(max_score, score[i]);
                }
                double sum = 0.0;
                for (size_t i = 0; i < keys.size(); ++i) {
                    score[i] = std::exp(score[i] - max_score);
                    sum += score[i];
                }
                for (int d = 0; d < test.head_dim; ++d) {
                    double acc = 0.0;
                    for (size_t i = 0; i < keys.size(); ++i) {
                        acc += score[i] *
                            ggml_fp16_to_fp32(test.v[kv_index(test, b, keys[i], kv_head, d)]);
                    }
                    (*out)[packed_index(test, b, token, head, d)] =
                        static_cast<float>(acc / sum);
                }
                (*out)[packed_index(test, b, token, head, test.head_dim)] =
                    static_cast<float>(max_score + std::log(sum));
            }
        }
    }
}

// Deterministic payload plus the CPU references for both branches.
void fill_case(attention_case & test) {
    test.mode = mask_mode_from_env();
    test.q.resize(static_cast<size_t>(test.batch) * test.tokens * test.heads * test.head_dim);
    test.k.resize(static_cast<size_t>(test.batch) * test.tokens * test.kv_heads * test.head_dim);
    test.v.resize(test.k.size());
    // Only the real joint tokens carry data.  The trailing block padding stays
    // zero, which is what the packed production buffers are padded with, so the
    // CPU reference and both kernels see the same padded block.
    const int real = real_tokens(test);
    for (int b = 0; b < test.batch; ++b) {
        for (int token = 0; token < real; ++token) {
            for (int head = 0; head < test.heads; ++head) {
                for (int d = 0; d < test.head_dim; ++d) {
                    const float phase = static_cast<float>(1 + b * 131 + token * 17 + head * 29 + d * 7);
                    test.q[q_index(test, b, token, head, d)] =
                        std::sin(phase * 0.0137f) * 1.75f;
                }
            }
            for (int head = 0; head < test.kv_heads; ++head) {
                for (int d = 0; d < test.head_dim; ++d) {
                    const float phase = static_cast<float>(3 + b * 97 + token * 19 + head * 31 + d * 11);
                    test.k[kv_index(test, b, token, head, d)] =
                        ggml_fp32_to_fp16(std::cos(phase * 0.0091f) * 0.85f);
                    test.v[kv_index(test, b, token, head, d)] =
                        ggml_fp32_to_fp16(std::sin(phase * 0.0073f) * 0.35f);
                }
            }
        }
    }

    fill_block_index(test);

    const float scale = 1.0f / std::sqrt(static_cast<float>(test.head_dim));
    test.reference.assign(packed_rows(test) * (test.head_dim + 1), 0.0f);
    std::vector<double> score(test.tokens, 0.0);
    for (int b = 0; b < test.batch; ++b) {
        for (int token = 0; token < test.tokens; ++token) {
            for (int head = 0; head < test.heads; ++head) {
                const int kv_head = head / (test.heads / test.kv_heads);
                double max_score = -1e30;
                for (int key = 0; key < test.tokens; ++key) {
                    double dot = 0.0;
                    for (int d = 0; d < test.head_dim; ++d) {
                        dot += static_cast<double>(test.q[q_index(test, b, token, head, d)]) *
                            ggml_fp16_to_fp32(test.k[kv_index(test, b, key, kv_head, d)]);
                    }
                    score[key] = dot * scale;
                    max_score = std::max(max_score, score[key]);
                }
                double sum = 0.0;
                for (int key = 0; key < test.tokens; ++key) {
                    score[key] = std::exp(score[key] - max_score);
                    sum += score[key];
                }
                for (int d = 0; d < test.head_dim; ++d) {
                    double acc = 0.0;
                    for (int key = 0; key < test.tokens; ++key) {
                        acc += score[key] *
                            ggml_fp16_to_fp32(test.v[kv_index(test, b, key, kv_head, d)]);
                    }
                    test.reference[packed_index(test, b, token, head, d)] =
                        static_cast<float>(acc / sum);
                }
                test.reference[packed_index(test, b, token, head, test.head_dim)] =
                    static_cast<float>(max_score + std::log(sum));
            }
        }
    }

    if (test.mode == mask_mode::all_kept) {
        // Keeping every block makes the masked attention the full attention, so
        // the second pass would be pure waste.
        test.sparse_reference = test.reference;
    } else {
        compute_reference(test, &test.mask, &test.sparse_reference);
    }
}

// ---------------------------------------------------------------------------
// Line fixture
// ---------------------------------------------------------------------------

struct line_spec {
    const char * name = "";
    // Emulates a device without rpcmem sharing: the OpenCL sparse branch owns a
    // second, independent rpcmem allocation, so the payload has to be copied
    // into it before the sparse branch runs and the result copied back into the
    // NPU allocation afterwards.
    bool separate_memory = false;
    // A single plain enqueue on the calling thread, with no submission queue.
    bool async = false;
    // Early submission from the sidecar submission thread.
    bool submit_thread = false;
    // Finishes the residual with the OpenCL exact-rho kernel.
    bool residual_on_opencl = false;
    // Route 1: the sparse branch is an HTP node as well, computed on the NPU
    // right after the dense node, so no OpenCL work happens on this line.
    bool sparse_on_npu = false;
    // Which line of the timeline figure the line stands for.
    const char * route = "";
};

struct line_fixture {
    ggml_context * ctx = nullptr;      // NPU side rpcmem tensors
    ggml_context * cl_ctx = nullptr;   // OpenCL side rpcmem tensors (separate memory lines)
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_buffer_t cl_buffer = nullptr;
    ggml_tensor * q_storage = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * mask = nullptr;
    // f16 attention mask of the NPU dense branch (the dense kernel route).
    ggml_tensor * dense_mask = nullptr;
    ggml_tensor * q_cl = nullptr;
    ggml_tensor * k_cl = nullptr;
    ggml_tensor * v_cl = nullptr;
    ggml_tensor * mask_cl = nullptr;
    ggml_tensor * dense = nullptr;
    ggml_tensor * sparse = nullptr;
    // Packed sparse result the residual reads inside the NPU allocation: the
    // sparse node output itself when both branches share one allocation, the
    // copy-back destination when they do not.
    ggml_tensor * sparse_result = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * rho = nullptr;
    ggml_backend_htp_async_cache_slot * residual_slot = nullptr;
    ggml_backend_htp_async_cache_slot * rho_slot = nullptr;
    ggml_cgraph * dense_graph = nullptr;
    ggml_cgraph * sidecar_graph = nullptr;
    // Route 1 keeps the sparse node on the NPU, so it gets its own HTP graph.
    ggml_cgraph * sparse_graph = nullptr;
    // Production names: the sidecar residual route tags the pair by tensor name
    // and requires the persistent import key to match it.
    const char * residual_key = "htp_block_sparse_residual_b0_m0_s0";
    const char * rho_key = "htp_block_sparse_residual_rho_b0_m0_s0";
    bool separate_memory = false;
};

void free_line_fixture(line_fixture & fx) {
    if (fx.residual_slot != nullptr) {
        ggml_backend_htp_async_cache_slot_free(fx.residual_slot);
    }
    if (fx.rho_slot != nullptr) {
        ggml_backend_htp_async_cache_slot_free(fx.rho_slot);
    }
    if (fx.cl_buffer != nullptr) {
        ggml_backend_buffer_free(fx.cl_buffer);
    }
    if (fx.buffer != nullptr) {
        ggml_backend_buffer_free(fx.buffer);
    }
    if (fx.cl_ctx != nullptr) {
        ggml_free(fx.cl_ctx);
    }
    if (fx.ctx != nullptr) {
        ggml_free(fx.ctx);
    }
    fx = {};
}

// Copies one tensor into a second rpcmem allocation through the mapped base
// pointers: the transfer a board without rpcmem sharing has to perform.
bool transfer_between_rpcmem(ggml_tensor * dst, const ggml_tensor * src) {
    if (dst == nullptr || src == nullptr) {
        return false;
    }
    const size_t size = ggml_nbytes(src);
    if (size == 0 || ggml_nbytes(dst) != size ||
        !ggml_is_contiguous(dst) || !ggml_is_contiguous(src) ||
        dst->data == nullptr || src->data == nullptr) {
        return false;
    }
    std::memcpy(dst->data, src->data, size);
    return true;
}

bool build_line_fixture(
        ggml_backend_t backend,
        const attention_case & test,
        const line_spec & spec,
        uint64_t generation,
        line_fixture & fx) {
    char reason[k_reason_capacity] = {};
    fx.separate_memory = spec.separate_memory;

    const ggml_init_params params = {64 * 1024 * 1024, nullptr, true};
    fx.ctx = ggml_init(params);
    if (fx.ctx == nullptr) {
        return false;
    }

    fx.q_storage = ggml_new_tensor_4d(
        fx.ctx, GGML_TYPE_F32, test.head_dim, test.heads, test.tokens_alloc, test.batch);
    fx.k_storage = ggml_new_tensor_4d(
        fx.ctx, GGML_TYPE_F16, test.head_dim, test.kv_heads, test.tokens_alloc, test.batch);
    fx.v_storage = ggml_new_tensor_4d(
        fx.ctx, GGML_TYPE_F16, test.head_dim, test.kv_heads, test.tokens_alloc, test.batch);
    fx.mask = ggml_new_tensor_2d(
        fx.ctx, GGML_TYPE_I32, q_blocks(test) + 1, test.heads * q_blocks(test));
    // NPU dense branch: a plain f16 attention mask, the kv extent padded to a
    // multiple of 64 exactly like the kernel benchmarks build it.
    fx.dense_mask = ggml_new_tensor_2d(
        fx.ctx, GGML_TYPE_F16, ((test.tokens + 63) / 64) * 64, test.tokens);

    if (spec.separate_memory) {
        fx.cl_ctx = ggml_init(params);
        if (fx.cl_ctx == nullptr) {
            return false;
        }
        fx.q_cl = ggml_new_tensor_4d(
            fx.cl_ctx, GGML_TYPE_F32, test.head_dim, test.heads, test.tokens_alloc, test.batch);
        fx.k_cl = ggml_new_tensor_4d(
            fx.cl_ctx, GGML_TYPE_F16, test.head_dim, test.kv_heads, test.tokens_alloc, test.batch);
        fx.v_cl = ggml_new_tensor_4d(
            fx.cl_ctx, GGML_TYPE_F16, test.head_dim, test.kv_heads, test.tokens_alloc, test.batch);
        fx.mask_cl = ggml_new_tensor_2d(
            fx.cl_ctx, GGML_TYPE_I32, q_blocks(test) + 1, test.heads * q_blocks(test));
        fx.sparse_result = ggml_new_tensor_4d(
            fx.ctx, GGML_TYPE_F32, test.head_dim + 1, test.heads, test.tokens, test.batch);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(test.head_dim));

    // The image token grid the static halo metadata describes:
    // image_block_rows * image_block_cols * k_block == image_tokens, followed by
    // the protected text context that closes the joint sequence.
    const int image_block_rows = test.image_block_rows;
    const int image_block_cols = test.image_block_cols;

    // NPU main path: the dense attention node on the route the HTP kernel
    // benchmark measures ("Dense"): an f16 attention mask with every real key
    // visible, no block index and no halo policy.
    {
        ggml_tensor * q = ggml_permute(fx.ctx, logical_tokens_view(fx.ctx, fx.q_storage, test), 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(fx.ctx, logical_tokens_view(fx.ctx, fx.k_storage, test), 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(fx.ctx, logical_tokens_view(fx.ctx, fx.v_storage, test), 0, 2, 1, 3);
        fx.dense = ggml_flash_attn_ext_with_lse(
            fx.ctx, q, k, v, fx.dense_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fx.dense, GGML_PREC_F32);
        ggml_set_name(fx.dense, "branch_dense_attention");
    }

    // OpenCL sparse branch.  Shared memory keeps it on the NPU allocation,
    // separate memory moves it, together with its Q/K/V/mask, to the second one.
    {
        ggml_context * sparse_ctx = spec.separate_memory ? fx.cl_ctx : fx.ctx;
        ggml_tensor * q_src = spec.separate_memory ? fx.q_cl : fx.q_storage;
        ggml_tensor * k_src = spec.separate_memory ? fx.k_cl : fx.k_storage;
        ggml_tensor * v_src = spec.separate_memory ? fx.v_cl : fx.v_storage;
        ggml_tensor * mask_src = spec.separate_memory ? fx.mask_cl : fx.mask;
        ggml_tensor * q = ggml_permute(sparse_ctx, logical_tokens_view(sparse_ctx, q_src, test), 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(sparse_ctx, logical_tokens_view(sparse_ctx, k_src, test), 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(sparse_ctx, logical_tokens_view(sparse_ctx, v_src, test), 0, 2, 1, 3);

        fx.sparse = ggml_flash_attn_ext_with_lse(
            sparse_ctx, q, k, v, mask_src, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_block_sparse(
            fx.sparse, k_block, k_block, GGML_FLASH_ATTN_EXT_MASK_BLOCK_INDEX);
        ggml_flash_attn_ext_set_static_halo_2d(
            fx.sparse, test.image_tokens, test.text_tokens, image_block_rows, image_block_cols);
        ggml_flash_attn_ext_set_prec(fx.sparse, GGML_PREC_F32);
        ggml_set_name(fx.sparse, "branch_sparse_attention");
    }

    // Residual destination and rho.  The OpenCL line uses the same rpcmem cache
    // slots production uses; the CPU lines only need plain rpcmem storage.
    if (spec.residual_on_opencl) {
        const int64_t residual_ne[GGML_MAX_DIMS] = {
            test.head_dim, test.heads, test.tokens, test.batch,
        };
        const int64_t rho_ne[GGML_MAX_DIMS] = {1, test.heads, test.tokens, test.batch};
        fx.residual_slot = ggml_backend_htp_async_cache_slot_create(
            backend, fx.residual_key, generation, GGML_TYPE_F16, residual_ne,
            reason, sizeof(reason));
        fx.rho_slot = ggml_backend_htp_async_cache_slot_create(
            backend, fx.rho_key, generation, GGML_TYPE_F32, rho_ne,
            reason, sizeof(reason));
        if (fx.residual_slot == nullptr || fx.rho_slot == nullptr) {
            std::fprintf(stderr, "FAIL: residual cache slots: %s\n", reason);
            return false;
        }
        fx.residual = ggml_backend_htp_async_cache_slot_tensor(fx.residual_slot);
        fx.rho = ggml_backend_htp_async_cache_slot_tensor(fx.rho_slot);
        fx.residual->op = GGML_OP_CUSTOM;
        fx.residual->src[0] = fx.dense;
        fx.residual->src[1] = fx.sparse;
        fx.residual->src[2] = fx.rho;
        fx.rho->op = GGML_OP_CUSTOM;
        fx.rho->src[0] = fx.dense;
        fx.rho->src[1] = fx.sparse;
    } else {
        fx.residual = ggml_new_tensor_4d(
            fx.ctx, GGML_TYPE_F16, test.head_dim, test.heads, test.tokens, test.batch);
        fx.rho = ggml_new_tensor_4d(
            fx.ctx, GGML_TYPE_F32, 1, test.heads, test.tokens, test.batch);
    }

    fx.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        fx.ctx, ggml_backend_htp_qkv_buffer_type());
    if (fx.buffer == nullptr) {
        std::fprintf(stderr, "FAIL: failed to allocate the NPU rpcmem tensors\n");
        return false;
    }
    if (spec.separate_memory) {
        fx.cl_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            fx.cl_ctx, ggml_backend_htp_qkv_buffer_type());
        if (fx.cl_buffer == nullptr) {
            std::fprintf(stderr, "FAIL: failed to allocate the OpenCL rpcmem tensors\n");
            return false;
        }
    }

    // The block padding past the real joint sequence must not carry stale
    // rpcmem, so the storages start blank and the real rows are written after.
    const auto blank = [](ggml_tensor * tensor) {
        std::vector<uint8_t> zeros(ggml_nbytes(tensor), 0);
        ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
    };
    blank(fx.q_storage);
    blank(fx.k_storage);
    blank(fx.v_storage);

    ggml_backend_tensor_set(fx.q_storage, test.q.data(), 0, test.q.size() * sizeof(float));
    ggml_backend_tensor_set(fx.k_storage, test.k.data(), 0, test.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(fx.v_storage, test.v.data(), 0, test.v.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(fx.mask, test.mask.data(), 0, test.mask.size() * sizeof(int32_t));
    {
        // f16 dense mask: zero for every real key token, -inf for the padding
        // the kv extent is rounded up to (the benchmark builds it the same way).
        const int padded_kv = static_cast<int>(fx.dense_mask->ne[0]);
        std::vector<ggml_fp16_t> dense_mask(
            static_cast<size_t>(padded_kv) * static_cast<size_t>(test.tokens),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int token = 0; token < test.tokens; ++token) {
            std::fill_n(dense_mask.data() + static_cast<size_t>(token) * padded_kv,
                        static_cast<size_t>(test.tokens), ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(fx.dense_mask, dense_mask.data(), 0,
                                dense_mask.size() * sizeof(ggml_fp16_t));
    }

    if (spec.separate_memory) {
        // The OpenCL allocation starts blank, so a missing copy cannot go unnoticed.
        blank(fx.q_cl);
        blank(fx.k_cl);
        blank(fx.v_cl);
        blank(fx.mask_cl);
    }

    fx.dense_graph = ggml_new_graph_custom(fx.ctx, 16, false);
    ggml_build_forward_expand(fx.dense_graph, fx.dense);
    if (spec.sparse_on_npu) {
        // Route 1: the sparse node is a plain HTP node inside the shared rpcmem
        // allocation, so it needs a graph of its own and no sidecar at all.
        fx.sparse_graph = ggml_new_graph_custom(fx.ctx, 16, false);
        ggml_build_forward_expand(fx.sparse_graph, fx.sparse);
        return true;
    }
    // The two branches live in two graphs: the NPU dense graph never sees the
    // OpenCL node, and the sidecar graph never sees the dense node (the sidecar
    // only routes the sparse producer).  The residual pair is appended by hand
    // for the same reason: expanding it would drag the dense node in and the
    // preflight would route it to OpenCL as well.
    fx.sidecar_graph = ggml_new_graph_custom(fx.ctx, 32, false);
    ggml_build_forward_expand(fx.sidecar_graph, fx.sparse);
    if (spec.residual_on_opencl) {
        ggml_graph_add_node(fx.sidecar_graph, fx.rho);
        ggml_graph_add_node(fx.sidecar_graph, fx.residual);
    }
    return true;
}

// ---------------------------------------------------------------------------
// rpcmem import descriptors
// ---------------------------------------------------------------------------

using descriptor_list = std::vector<ggml_opencl_htp_import_descriptor>;

bool collect_descriptors(const line_fixture & fx, descriptor_list * descriptors) {
    const auto add = [&](ggml_backend_buffer_t backing,
                         const ggml_backend_htp_rpcmem_info & info) {
        for (const ggml_opencl_htp_import_descriptor & existing : *descriptors) {
            if (existing.backing == backing) {
                return;
            }
        }
        descriptors->push_back(
            {backing, info.base, info.fd, info.logical_size, info.allocation_size});
    };

    ggml_backend_buffer_t backings[3] = {fx.buffer, fx.cl_buffer, nullptr};
    for (size_t i = 0; backings[i] != nullptr; ++i) {
        ggml_backend_htp_rpcmem_info info = {};
        if (!ggml_backend_htp_get_rpcmem_info(backings[i], &info)) {
            std::fprintf(stderr, "FAIL: rpcmem info unavailable for backing %p\n",
                         static_cast<void *>(backings[i]));
            return false;
        }
        add(backings[i], info);
    }

    ggml_backend_htp_async_cache_slot * slots[3] = {fx.residual_slot, fx.rho_slot, nullptr};
    for (size_t i = 0; slots[i] != nullptr; ++i) {
        ggml_backend_htp_rpcmem_info info = {};
        if (!ggml_backend_htp_async_cache_slot_rpcmem_info(slots[i], &info)) {
            std::fprintf(stderr, "FAIL: cache slot rpcmem info unavailable\n");
            return false;
        }
        add(ggml_backend_htp_async_cache_slot_tensor(slots[i])->buffer, info);
    }
    return true;
}

const ggml_opencl_htp_import_descriptor * find_descriptor(
        const descriptor_list & descriptors, ggml_backend_buffer_t backing) {
    for (const ggml_opencl_htp_import_descriptor & descriptor : descriptors) {
        if (descriptor.backing == backing) {
            return &descriptor;
        }
    }
    return nullptr;
}

bool acquire_persistent(
        ggml_opencl_htp_sidecar * sidecar,
        const descriptor_list & descriptors,
        ggml_backend_buffer_t backing,
        const char * key,
        uint64_t generation) {
    const ggml_opencl_htp_import_descriptor * descriptor =
        find_descriptor(descriptors, backing);
    if (descriptor == nullptr) {
        std::fprintf(stderr, "FAIL: no rpcmem descriptor for backing %p\n",
                     static_cast<void *>(backing));
        return false;
    }
    char reason[k_reason_capacity] = {};
    uint64_t import_identity = 0;
    return check_result(
        ggml_backend_opencl_htp_persistent_import_acquire(
            sidecar, descriptor, key, generation, &import_identity, reason, sizeof(reason)),
        reason, "failed to acquire a persistent rpcmem import");
}

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------

// Production semantics: rho = exp(lse_sparse - lse_dense) and
// residual[d] = dense_o[d] - rho * sparse_o[d] over packed rows.
void compute_cpu_residual(
        const attention_case & test,
        const std::vector<float> & dense,
        const std::vector<float> & sparse,
        std::vector<ggml_fp16_t> * residual,
        std::vector<float> * rho) {
    const size_t rows = packed_rows(test);
    const int head_dim = test.head_dim;
    residual->assign(rows * static_cast<size_t>(head_dim), ggml_fp32_to_fp16(0.0f));
    rho->assign(rows, 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed = row * static_cast<size_t>(head_dim + 1);
        const double value = std::exp(
            static_cast<double>(sparse[packed + head_dim]) -
            static_cast<double>(dense[packed + head_dim]));
        (*rho)[row] = static_cast<float>(value);
        for (int d = 0; d < head_dim; ++d) {
            (*residual)[row * static_cast<size_t>(head_dim) + static_cast<size_t>(d)] =
                ggml_fp32_to_fp16(static_cast<float>(
                    static_cast<double>(dense[packed + d]) - value * sparse[packed + d]));
        }
    }
}

double compare_packed(
        const std::vector<float> & actual,
        const std::vector<float> & reference,
        int head_dim) {
    double max_abs = 0.0;
    const size_t rows = actual.size() / static_cast<size_t>(head_dim + 1);
    for (size_t row = 0; row < rows; ++row) {
        for (int d = 0; d <= head_dim; ++d) {
            const size_t index = row * (head_dim + 1) + static_cast<size_t>(d);
            max_abs = std::max(
                max_abs,
                std::fabs(static_cast<double>(actual[index]) -
                          static_cast<double>(reference[index])));
        }
    }
    return max_abs;
}

double compare_fp16(
        const std::vector<ggml_fp16_t> & actual,
        const std::vector<ggml_fp16_t> & reference) {
    double max_abs = 0.0;
    const size_t count = std::min(actual.size(), reference.size());
    for (size_t i = 0; i < count; ++i) {
        max_abs = std::max(
            max_abs,
            std::fabs(static_cast<double>(ggml_fp16_to_fp32(actual[i])) -
                      static_cast<double>(ggml_fp16_to_fp32(reference[i]))));
    }
    return max_abs;
}

// ---------------------------------------------------------------------------
// Stage timeline
// ---------------------------------------------------------------------------

// The figure's legend: which resource a drawn box belongs to.
enum class stage_kind { host, npu, gpu, waiting, others };

const char * stage_kind_name(stage_kind kind) {
    switch (kind) {
        case stage_kind::host:    return "host";
        case stage_kind::npu:     return "npu";
        case stage_kind::gpu:     return "gpu";
        case stage_kind::waiting: return "waiting";
        case stage_kind::others:  return "others";
    }
    return "others";
}

char stage_kind_glyph(stage_kind kind) {
    switch (kind) {
        case stage_kind::host:    return 'H';
        case stage_kind::npu:     return 'N';
        case stage_kind::gpu:     return 'G';
        case stage_kind::waiting: return 'W';
        case stage_kind::others:  return '.';
    }
    return '?';
}

// The figure draws two rows: the main path and the GPU branch.
enum class stage_lane { main, gpu };

const char * stage_lane_name(stage_lane lane) {
    return lane == stage_lane::main ? "main" : "gpu";
}

// One drawn box of the pair timeline.  `name` is a string literal owned by the
// test, so a timeline can be copied and averaged by value.
struct stage_span {
    const char * name = "";
    stage_kind kind = stage_kind::others;
    stage_lane lane = stage_lane::main;
    double begin_ms = 0.0;
    double end_ms = 0.0;
};

double stage_span_ms(const stage_span & span) {
    return span.end_ms > span.begin_ms ? span.end_ms - span.begin_ms : 0.0;
}

// Draw order: main path first, then the GPU branch, each in time order, which
// is how the figure lays the pair out.  Applied to a closed timeline and again
// after the spans are averaged, because an average can move a span (the grey
// box) across its neighbours.
void sort_stage_spans(std::vector<stage_span> & spans) {
    std::stable_sort(spans.begin(), spans.end(),
                     [](const stage_span & a, const stage_span & b) {
                         if (a.lane != b.lane) {
                             return a.lane == stage_lane::main;
                         }
                         return a.begin_ms < b.begin_ms;
                     });
}

// The stages of one attention pair, in the order the main path and the GPU
// branch execute them.  Spans never overlap, so the pair window can be drawn
// straight from them.
struct stage_timeline {
    std::vector<stage_span> spans;

    void record(const char * name, stage_kind kind, stage_lane lane,
                int64_t begin_us, int64_t end_us, int64_t origin_us) {
        if (end_us < begin_us) {
            end_us = begin_us;
        }
        stage_span span;
        span.name = name;
        span.kind = kind;
        span.lane = lane;
        span.begin_ms = static_cast<double>(begin_us - origin_us) / 1000.0;
        span.end_ms = static_cast<double>(end_us - origin_us) / 1000.0;
        spans.push_back(span);
    }

    bool has_lane(stage_lane lane) const {
        for (const stage_span & span : spans) {
            if (span.lane == lane) {
                return true;
            }
        }
        return false;
    }

    double duration_of(stage_kind kind, stage_lane lane) const {
        double sum = 0.0;
        for (const stage_span & span : spans) {
            if (span.kind == kind && span.lane == lane) {
                sum += stage_span_ms(span);
            }
        }
        return sum;
    }

    // The two lanes run concurrently, so a lane is measured on its own: the
    // main path spans tile the pair window, the GPU branch does not have to.
    double measured_ms(stage_lane lane) const {
        double sum = 0.0;
        for (const stage_span & span : spans) {
            if (span.lane == lane) {
                sum += stage_span_ms(span);
            }
        }
        return sum;
    }

    // Appends the grey "others" box, the figure's "computation between two
    // attentions" plus the host bookkeeping this test does not name: the main
    // path time no measured span covers.  The box is drawn over the widest gap
    // between the measured spans, which is where the main path really idles,
    // and is capped at that gap so it never overlaps a measured span; the tail
    // of the window is the fallback when the idle is spread over many small
    // gaps.  others_ms is the drawn box, so an idle a run spreads over several
    // gaps shows up as the box being shorter than the idle.
    void close(double pair_ms) {
        double cursor = 0.0;
        double widest_gap = 0.0;
        double widest_begin = 0.0;
        for (const stage_span & span : spans) {
            if (span.lane != stage_lane::main) {
                continue;
            }
            if (span.begin_ms > cursor && span.begin_ms - cursor > widest_gap) {
                widest_gap = span.begin_ms - cursor;
                widest_begin = cursor;
            }
            cursor = std::max(cursor, span.end_ms);
        }
        if (pair_ms - cursor > widest_gap) {
            widest_gap = pair_ms - cursor;
            widest_begin = cursor;
        }
        const double leftover = pair_ms - measured_ms(stage_lane::main);
        if (leftover > 1e-9) {
            stage_span span;
            span.name = "others";
            span.kind = stage_kind::others;
            span.lane = stage_lane::main;
            if (widest_gap > 1e-9) {
                span.begin_ms = widest_begin;
                span.end_ms = widest_begin + std::min(leftover, widest_gap);
            } else {
                span.end_ms = pair_ms;
                span.begin_ms = pair_ms - leftover;
            }
            spans.push_back(span);
        }
        sort_stage_spans(spans);
    }
};

// The dispatch window of an HTP node and the device time it reports.  Only the
// total window and the device time are observable, so without a doorbell the
// compute span is pinned to the end of the window and everything in front of it
// is the single host box.
int64_t htp_compute_begin_us(int64_t dispatch_begin_us, int64_t dispatch_end_us,
                             double reported_ms) {
    const int64_t reported_us =
        static_cast<int64_t>(std::max(0.0, reported_ms) * 1000.0);
    return std::max(dispatch_begin_us, dispatch_end_us - reported_us);
}

// The three boxes of a dispatched HTP node: the host window in front of the DSP
// doorbell (submit), the device time the node reports (compute) and the host
// tail behind it (sync).  The doorbell comes from
// ggml_backend_htp_get_last_dispatch_doorbell_us; without it (no DSP dispatch,
// no hook) the host window stays merged and `split` is false.
struct htp_node_boxes {
    bool split = false;
    int64_t submit_end_us = 0;
    int64_t compute_end_us = 0;
};

htp_node_boxes htp_node_boxes_for(int64_t dispatch_begin_us, int64_t dispatch_end_us,
                                  double reported_ms, int64_t doorbell_us) {
    htp_node_boxes boxes;
    if (doorbell_us > dispatch_begin_us && doorbell_us < dispatch_end_us) {
        boxes.split = true;
        boxes.submit_end_us = doorbell_us;
        boxes.compute_end_us = std::min(
            dispatch_end_us,
            doorbell_us + static_cast<int64_t>(std::max(0.0, reported_ms) * 1000.0));
    } else {
        boxes.submit_end_us =
            htp_compute_begin_us(dispatch_begin_us, dispatch_end_us, reported_ms);
        boxes.compute_end_us = dispatch_end_us;
    }
    return boxes;
}

// The CL event timestamps are on the host's monotonic clock when the driver
// uses the same base, which Qualcomm's does.  Anything more than a second apart
// means the two clocks are unrelated, and the caller falls back to placing the
// kernel right after the submit call.
bool cl_event_on_host_clock(int64_t wall_us, uint64_t event_ns) {
    const int64_t delta_ns = wall_us * 1000 - static_cast<int64_t>(event_ns);
    return delta_ns > -1000000000LL && delta_ns < 1000000000LL;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct line_metrics {
    // Everything below is an input to the stage boxes, not a reported column.
    // CL event timestamps of the sparse kernel, the main-path submit call of
    // that dispatch, and the CPU residual window place the `sparse compute`,
    // `submit sparse` and `cpu residual compute` spans.
    double gpu_kernel_ms = 0.0;
    double sparse_submit_ms = 0.0;
    double residual_compute_ms = 0.0;
    // The explicit rpcmem <-> rpcmem transfers: what the test reports as the
    // copy amount and the copy time.
    double copy_in_ms = 0.0;
    double copy_out_ms = 0.0;
    size_t copy_in_bytes = 0;
    size_t copy_out_bytes = 0;
    // Start of the pair -> residual ready: the window the stage timeline is
    // measured against and reported as `pair_ms`.
    double residual_ms = 0.0;
    // Which line of the timeline figure this is.
    const char * route = "";
    // The stages of the pair, in execution order.
    stage_timeline timeline;
};

double mean_of(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

// A line runs the same stages every time, so the mean timeline averages each
// stage over the repeats.  The stages are matched by name, not by position: the
// only span whose place moves between runs is the grey box, which is drawn in
// whichever host gap happens to be widest that run, and matching by index would
// throw the whole timeline away (and with it the per-kind sums, which would then
// describe the first run while pair_ms describes the mean) as soon as it moves.
// A run that is missing a stage or has an extra one (a failing line, an early
// return) leaves the first run's timeline in place rather than inventing a blend
// of two different shapes.
stage_timeline summarise_timeline(const std::vector<line_metrics> & runs) {
    stage_timeline summary = runs.front().timeline;
    if (runs.size() < 2) {
        return summary;
    }
    const size_t span_count = summary.spans.size();
    std::vector<double> begin(span_count, 0.0);
    std::vector<double> end(span_count, 0.0);
    for (const line_metrics & run : runs) {
        if (run.timeline.spans.size() != span_count) {
            return summary;
        }
        std::vector<bool> matched(span_count, false);
        for (const stage_span & span : run.timeline.spans) {
            size_t slot = span_count;
            for (size_t i = 0; i < span_count; ++i) {
                if (matched[i]) {
                    continue;
                }
                if (summary.spans[i].kind == span.kind &&
                    summary.spans[i].lane == span.lane &&
                    std::strcmp(summary.spans[i].name, span.name) == 0) {
                    slot = i;
                    break;
                }
            }
            if (slot == span_count) {
                return summary;
            }
            matched[slot] = true;
            begin[slot] += span.begin_ms;
            end[slot] += span.end_ms;
        }
    }
    const double count = static_cast<double>(runs.size());
    for (size_t i = 0; i < span_count; ++i) {
        summary.spans[i].begin_ms = begin[i] / count;
        summary.spans[i].end_ms = end[i] / count;
    }
    sort_stage_spans(summary.spans);
    return summary;
}

// Summarises one line over all repeats.  The stages are averaged (see
// summarise_timeline) and so are the copy numbers; the copy amount is the same
// on every repeat, so the first run's byte counts stand.
line_metrics summarise_line(const std::vector<line_metrics> & runs) {
    line_metrics summary;
    if (runs.empty()) {
        return summary;
    }
    std::vector<double> gpu_kernel, sparse_submit, residual_compute;
    std::vector<double> copy_in, copy_out, residual;
    for (const line_metrics & run : runs) {
        gpu_kernel.push_back(run.gpu_kernel_ms);
        sparse_submit.push_back(run.sparse_submit_ms);
        residual_compute.push_back(run.residual_compute_ms);
        copy_in.push_back(run.copy_in_ms);
        copy_out.push_back(run.copy_out_ms);
        residual.push_back(run.residual_ms);
    }
    summary.gpu_kernel_ms = mean_of(gpu_kernel);
    summary.sparse_submit_ms = mean_of(sparse_submit);
    summary.residual_compute_ms = mean_of(residual_compute);
    summary.copy_in_ms = mean_of(copy_in);
    summary.copy_out_ms = mean_of(copy_out);
    summary.residual_ms = mean_of(residual);
    summary.copy_in_bytes = runs.front().copy_in_bytes;
    summary.copy_out_bytes = runs.front().copy_out_bytes;
    summary.route = runs.front().route;
    summary.timeline = summarise_timeline(runs);
    return summary;
}

// One drawn box of the pair timeline.
void print_stage(
        const char * name,
        const char * tag,
        const line_metrics & metrics,
        const stage_span & span) {
    const double duration = stage_span_ms(span);
    const double share = metrics.residual_ms > 0.0 ? duration / metrics.residual_ms : 0.0;
    std::printf(
        "stage line=%s tag=%s route=%s lane=%s name=\"%s\" kind=%s "
        "begin_ms=%.3f end_ms=%.3f dur_ms=%.3f share=%.4f\n",
        name, tag, metrics.route, stage_lane_name(span.lane), span.name,
        stage_kind_name(span.kind), span.begin_ms, span.end_ms, duration, share);
}

// Per-lane totals: the main path and the GPU branch overlap in time, so each
// row is summed on its own and the two rows are the two rows of the figure.
void print_stage_totals(const char * name, const char * tag, const line_metrics & metrics) {
    for (const stage_lane lane : {stage_lane::main, stage_lane::gpu}) {
        if (!metrics.timeline.has_lane(lane)) {
            continue;
        }
        std::printf(
            "stage_totals line=%s tag=%s route=%s lane=%s pair_ms=%.3f "
            "host_ms=%.3f npu_ms=%.3f gpu_ms=%.3f waiting_ms=%.3f others_ms=%.3f "
            "measured_ms=%.3f\n",
            name, tag, metrics.route, stage_lane_name(lane), metrics.residual_ms,
            metrics.timeline.duration_of(stage_kind::host, lane),
            metrics.timeline.duration_of(stage_kind::npu, lane),
            metrics.timeline.duration_of(stage_kind::gpu, lane),
            metrics.timeline.duration_of(stage_kind::waiting, lane),
            metrics.timeline.duration_of(stage_kind::others, lane),
            metrics.timeline.measured_ms(lane));
    }
}

// ASCII bar of one lane of the pair timeline, one column per percent of the
// pair, so the measured stages can be compared with the figure at a glance.
void print_stage_bar(
        const char * name,
        const char * tag,
        const line_metrics & metrics,
        stage_lane lane) {
    constexpr int k_columns = 100;
    const double pair_ms = metrics.residual_ms;
    std::string bar(static_cast<size_t>(k_columns), ' ');
    if (pair_ms > 0.0) {
        for (const stage_span & span : metrics.timeline.spans) {
            if (span.lane != lane || stage_span_ms(span) <= 0.0) {
                continue;
            }
            const double begin = std::max(0.0, std::min(pair_ms, span.begin_ms));
            const double end = std::max(0.0, std::min(pair_ms, span.end_ms));
            const int first = std::min(k_columns - 1,
                                       static_cast<int>(begin / pair_ms * k_columns));
            int last = std::max(first + 1,
                                static_cast<int>(std::ceil(end / pair_ms * k_columns)));
            last = std::min(k_columns, last);
            for (int column = first; column < last; ++column) {
                bar[static_cast<size_t>(column)] = stage_kind_glyph(span.kind);
            }
        }
    }
    std::printf("timeline line=%s tag=%s route=%s lane=%s pair_ms=%.3f |%s|\n",
                name, tag, metrics.route, stage_lane_name(lane), pair_ms, bar.c_str());
}

void print_stages(const char * name, const char * tag, const line_metrics & metrics) {
    for (const stage_span & span : metrics.timeline.spans) {
        print_stage(name, tag, metrics, span);
    }
    if (metrics.timeline.spans.empty()) {
        return;
    }
    print_stage_totals(name, tag, metrics);
    const char * timeline_env = std::getenv("HTP_OPENCL_BRANCH_TIMELINE");
    if (timeline_env == nullptr || timeline_env[0] != '0') {
        print_stage_bar(name, tag, metrics, stage_lane::main);
        if (metrics.timeline.has_lane(stage_lane::gpu)) {
            print_stage_bar(name, tag, metrics, stage_lane::gpu);
        }
    }
}

// The explicit copies of a line: the payload moved between the two rpcmem
// allocations and the result moved back, as an amount and a time each.  Lines
// without a second allocation report zeros.
void print_copy(const char * name, const char * tag, const line_metrics & metrics) {
    const double copy_in_mb =
        static_cast<double>(metrics.copy_in_bytes) / (1024.0 * 1024.0);
    const double copy_out_mb =
        static_cast<double>(metrics.copy_out_bytes) / (1024.0 * 1024.0);
    std::printf(
        "copy line=%s tag=%s route=%s "
        "copy_in_mb=%.3f copy_in_ms=%.3f copy_out_mb=%.3f copy_out_ms=%.3f "
        "copy_mb=%.3f copy_ms=%.3f\n",
        name, tag, metrics.route,
        copy_in_mb, metrics.copy_in_ms, copy_out_mb, metrics.copy_out_ms,
        copy_in_mb + copy_out_mb, metrics.copy_in_ms + metrics.copy_out_ms);
}

// One observation of a line: its stage timeline and its copies.
void print_line(const char * name, const line_metrics & metrics, const char * tag) {
    print_stages(name, tag, metrics);
    print_copy(name, tag, metrics);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Line environment
// ---------------------------------------------------------------------------

// Everything the sidecar reads has to be in place before the corresponding
// object exists: the HTP runtime builds its own sidecar from these flags, and
// this test's sidecar reads them again at graph create / preflight time.
void configure_common_route() {
    set_env("SD_HTP_OPS", "1");
    set_env("SD_HTP_COMPUTE", "1");
    set_env("SD_HTP_ATTENTION", "1");
    set_env("SD_HTP_BLOCK_SPARSE_ATTN", "1");
    set_env("SD_HTP_PACKED_ATTENTION", "0");
    set_env("SD_HTP_SPARSE_OPENCL", "1");
    set_env("SD_HTP_SPARSE_OPENCL_STRICT", "1");
    set_env("SD_HTP_SPARSE_OPENCL_PROFILE", "1");
    set_env("SD_HTP_SPARSE_OPENCL_LIFETIME_MODE", "deferred");
    set_env("SD_HTP_SPARSE_OPENCL_TRACE",
            std::getenv("HTP_OPENCL_BRANCH_TRACE") != nullptr ? "1" : "0");
    set_env("SD_HTP_HALO_KV_LOAD", "l2fetch");
    set_env("SD_HTP_BLOCK_SPARSE_PATCH_HALO", "1");
}

// ---------------------------------------------------------------------------
// One line
// ---------------------------------------------------------------------------

// The submission thread publishes the residual job asynchronously, so poll the
// cache generation until it is either queued or done.
bool wait_for_cache_publication(
        ggml_opencl_htp_sidecar * sidecar,
        const char * key,
        uint64_t generation,
        int timeout_ms) {
    const int64_t deadline_us = now_us() + static_cast<int64_t>(timeout_ms) * 1000;
    // The worker publishes within microseconds, so spin briefly instead of
    // quantising the residual latency with 1 ms sleeps.
    for (int spin = 0; now_us() < deadline_us; ++spin) {
        ggml_opencl_htp_cache_state state = GGML_OPENCL_HTP_CACHE_EMPTY;
        char reason[k_reason_capacity] = {};
        if (ggml_backend_opencl_htp_cache_query(
                sidecar, key, generation, &state, reason, sizeof(reason)) !=
            GGML_OPENCL_HTP_OK) {
            std::fprintf(stderr, "FAIL: cache query for %s: %s\n", key, reason);
            return false;
        }
        if (state == GGML_OPENCL_HTP_CACHE_SUBMITTED ||
            state == GGML_OPENCL_HTP_CACHE_READY) {
            return true;
        }
        if (state == GGML_OPENCL_HTP_CACHE_FAILED) {
            std::fprintf(stderr, "FAIL: cache job for %s failed\n", key);
            return false;
        }
        if (spin < 500) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    std::fprintf(stderr, "FAIL: residual job for %s was never submitted\n", key);
    return false;
}

int run_line(
        ggml_backend_t backend,
        ggml_opencl_htp_sidecar * sidecar,
        const attention_case & test,
        const line_spec & spec,
        uint64_t generation,
        line_metrics * metrics) {
    char reason[k_reason_capacity] = {};

    line_fixture fx;
    if (!check(build_line_fixture(backend, test, spec, generation, fx),
               "failed to build the line fixture")) {
        free_line_fixture(fx);
        return 2;
    }

    descriptor_list descriptors;
    if (!spec.sparse_on_npu) {
        if (!check(collect_descriptors(fx, &descriptors),
                   "failed to collect the rpcmem import descriptors")) {
            free_line_fixture(fx);
            return 2;
        }
    }

    // The tagged residual pair is only accepted when both cache slots are
    // already live persistent imports whose key matches the tensor name.
    bool residual_imported = false;
    bool rho_imported = false;
    if (spec.residual_on_opencl) {
        if (!acquire_persistent(
                sidecar, descriptors, fx.residual->buffer, fx.residual_key, generation) ||
            !acquire_persistent(
                sidecar, descriptors, fx.rho->buffer, fx.rho_key, generation)) {
            free_line_fixture(fx);
            return 2;
        }
        residual_imported = true;
        rho_imported = true;
    }

    // Route 1 has no sidecar graph at all: both branches are HTP nodes.
    ggml_opencl_htp_graph * graph = nullptr;
    if (!spec.sparse_on_npu) {
        const ggml_opencl_htp_result preflight = ggml_backend_opencl_htp_graph_preflight(
            sidecar, fx.sidecar_graph, descriptors.data(), descriptors.size(),
            &graph, reason, sizeof(reason));
        if (preflight != GGML_OPENCL_HTP_OK || graph == nullptr ||
            !ggml_backend_opencl_htp_graph_routes_node(graph, fx.sparse) ||
            ggml_backend_opencl_htp_graph_routes_node(graph, fx.dense) ||
            ggml_backend_opencl_htp_graph_routed_node_count(graph) == 0) {
            std::fprintf(stderr, "FAIL: sidecar preflight failed result=%d reason=%s\n",
                         static_cast<int>(preflight), reason[0] != '\0' ? reason : "unknown");
            if (graph != nullptr) {
                ggml_backend_opencl_htp_graph_free(&graph);
            }
            if (residual_imported) {
                ggml_backend_opencl_htp_persistent_import_release(
                    sidecar, fx.residual_key, generation, reason, sizeof(reason));
            }
            if (rho_imported) {
                ggml_backend_opencl_htp_persistent_import_release(
                    sidecar, fx.rho_key, generation, reason, sizeof(reason));
            }
            free_line_fixture(fx);
            return 2;
        }
    }

    int result = 0;
    do {
        if (spec.residual_on_opencl) {
            // The graph now owns the residual job and the submission thread:
            // preflight routed the sparse producer plus the tagged residual
            // pair, and the deferred residual job is published by the worker.
            if (!check_result(ggml_backend_opencl_htp_graph_start_submission_thread(
                                  graph, reason, sizeof(reason)),
                              reason, "failed to start the sidecar submission thread")) {
                result = 2;
                break;
            }
        }

        // The verification buffers are allocated before the measured window on
        // purpose: they are ~52 MB and touching them otherwise lands between
        // the dense branch and the residual and would show up as host time
        // inside the pair.
        ggml_tensor * sparse_consumer =
            fx.sparse_result != nullptr ? fx.sparse_result : fx.sparse;
        std::vector<float> dense_out(static_cast<size_t>(ggml_nelements(fx.dense)));
        std::vector<float> sparse_out(static_cast<size_t>(ggml_nelements(sparse_consumer)));
        std::vector<ggml_fp16_t> expected_residual;
        std::vector<float> expected_rho;
        double residual_max_abs = 0.0;
        // The rho the pair actually produced: the OpenCL exact-rho kernel output
        // on route 3, the host value the test stored itself on the CPU-residual
        // lines.
        std::vector<float> rho_out;

        // Start of the measured pair.
        const int64_t pair_begin_us = now_us();

        // Every stage of the pair is recorded as a span of this timeline, named
        // like the boxes of the figure.
        stage_timeline timeline;
        const auto record_stage = [&](const char * stage_name, stage_kind kind,
                                      stage_lane lane, int64_t begin_us, int64_t end_us) {
            timeline.record(stage_name, kind, lane, begin_us, end_us, pair_begin_us);
        };
        metrics->route = spec.route;

        // Explicit copy in: nothing but this move puts the payload into the
        // second rpcmem allocation.
        int64_t copy_in_us = 0;
        if (fx.separate_memory) {
            const int64_t begin = now_us();
            const bool moved =
                transfer_between_rpcmem(fx.q_cl, fx.q_storage) &&
                transfer_between_rpcmem(fx.k_cl, fx.k_storage) &&
                transfer_between_rpcmem(fx.v_cl, fx.v_storage) &&
                transfer_between_rpcmem(fx.mask_cl, fx.mask);
            copy_in_us = now_us() - begin;
            if (!check(moved, "failed to copy the payload into the OpenCL rpcmem")) {
                result = 2;
                break;
            }
            metrics->copy_in_bytes = ggml_nbytes(fx.q_storage) + ggml_nbytes(fx.k_storage) +
                ggml_nbytes(fx.v_storage) + ggml_nbytes(fx.mask);
        }
        metrics->copy_in_ms = static_cast<double>(copy_in_us) / 1000.0;
        if (copy_in_us > 0) {
            record_stage("copy in", stage_kind::host, stage_lane::main,
                         pair_begin_us, pair_begin_us + copy_in_us);
        }

        if (!spec.sparse_on_npu) {
            ggml_backend_opencl_set_attention_profiling_enabled(true);
            ggml_backend_opencl_reset_attention_profile();
            ggml_backend_htp_reset_attention_profile(backend);
        }

        // 1. Sparse submission.  Route 1 has no GPU branch: its sparse node is
        // an NPU node and is submitted after the dense node instead.
        int64_t submit_begin_us = 0;
        int64_t submit_end_us = 0;
        if (spec.sparse_on_npu) {
            // Nothing to submit yet.
        } else if (!spec.submit_thread) {
            submit_begin_us = now_us();
            const ggml_opencl_htp_result sparse_result =
                ggml_backend_opencl_htp_graph_compute_block_sparse_attn_async(
                    graph, fx.sparse, reason, sizeof(reason));
            submit_end_us = now_us();
            if (!check_result(sparse_result, reason,
                              "asynchronous OpenCL sparse attention failed")) {
                result = 1;
                break;
            }
        } else {
            submit_begin_us = now_us();
            const ggml_opencl_htp_result sparse_result =
                ggml_backend_opencl_htp_graph_queue_block_sparse_attn(
                    graph, fx.sparse, reason, sizeof(reason));
            submit_end_us = now_us();
            if (!check_result(sparse_result, reason,
                              "failed to queue the OpenCL sparse job")) {
                result = 1;
                break;
            }
        }
        metrics->sparse_submit_ms =
            static_cast<double>(submit_end_us - submit_begin_us) / 1000.0;

        // 2. NPU dense attention: the main path never blocks on the GPU here.
        const int64_t dense_begin_us = now_us();
        const ggml_status dense_status =
            ggml_backend_graph_compute(backend, fx.dense_graph);
        const int64_t dense_end_us = now_us();
        // DSP-reported time of the node, and the instant its doorbell was
        // rung: together they cut the dispatch window into submit / compute /
        // sync, which is what the three boxes of the figure are drawn from.
        const double dense_reported_ms =
            static_cast<double>(ggml_backend_htp_get_last_attention_elapsed_us(backend)) / 1000.0;
        if (dense_status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: NPU dense attention returned status=%d\n",
                         static_cast<int>(dense_status));
            result = 1;
            break;
        }
        const int64_t dense_doorbell_us =
            ggml_backend_htp_get_last_dispatch_doorbell_us(backend);
        {
            const htp_node_boxes boxes = htp_node_boxes_for(
                dense_begin_us, dense_end_us, dense_reported_ms, dense_doorbell_us);
            record_stage(boxes.split ? "submit dense" : "submit+sync dense",
                         stage_kind::host, stage_lane::main,
                         dense_begin_us, boxes.submit_end_us);
            record_stage("dense compute", stage_kind::npu, stage_lane::main,
                         boxes.submit_end_us, boxes.compute_end_us);
            if (boxes.compute_end_us < dense_end_us) {
                record_stage("sync dense", stage_kind::host, stage_lane::main,
                             boxes.compute_end_us, dense_end_us);
            }
        }

        // 2b. Route 1: the sparse node runs on the NPU as well, so the dense
        // and the sparse attention are strictly serial on the main path.
        if (spec.sparse_on_npu) {
            ggml_backend_htp_reset_attention_profile(backend);
            const int64_t sparse_begin_us = now_us();
            const ggml_status sparse_status =
                ggml_backend_graph_compute(backend, fx.sparse_graph);
            const int64_t sparse_end_us = now_us();
            const double sparse_reported_ms = static_cast<double>(
                ggml_backend_htp_get_last_attention_elapsed_us(backend)) / 1000.0;
            if (sparse_status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "FAIL: NPU sparse attention returned status=%d\n",
                             static_cast<int>(sparse_status));
                result = 1;
                break;
            }
            const int64_t sparse_doorbell_us =
                ggml_backend_htp_get_last_dispatch_doorbell_us(backend);
            const htp_node_boxes boxes = htp_node_boxes_for(
                sparse_begin_us, sparse_end_us, sparse_reported_ms, sparse_doorbell_us);
            record_stage(boxes.split ? "submit sparse" : "submit+sync sparse",
                         stage_kind::host, stage_lane::main,
                         sparse_begin_us, boxes.submit_end_us);
            record_stage("sparse compute", stage_kind::npu, stage_lane::main,
                         boxes.submit_end_us, boxes.compute_end_us);
            if (boxes.compute_end_us < sparse_end_us) {
                record_stage("sync sparse", stage_kind::host, stage_lane::main,
                             boxes.compute_end_us, sparse_end_us);
            }
        }

        // 3. Deferred wait for the sparse producer.
        int64_t wait_begin_us = 0;
        int64_t wait_end_us = 0;
        if (!spec.sparse_on_npu) {
            wait_begin_us = now_us();
            bool waited = false;
            const ggml_opencl_htp_result wait_result =
                ggml_backend_opencl_htp_graph_wait_for_tensor(
                    graph, fx.sparse, &waited, reason, sizeof(reason));
            wait_end_us = now_us();
            if (!check_result(wait_result, reason,
                              "failed to wait for the OpenCL sparse producer")) {
                result = 1;
                break;
            }
        }
        ggml_opencl_attention_profile profile = {};
        bool has_profile = false;
        if (!spec.sparse_on_npu) {
            has_profile = ggml_backend_opencl_get_last_attention_profile(&profile);
            ggml_backend_opencl_set_attention_profiling_enabled(false);
        }
        // The `sparse compute` box is the CL kernel itself, timed by the CL
        // event timestamps; without them it falls back to the host window of
        // the submit call, which is what the warning below says.
        if (has_profile) {
            if (profile.event_end_ns > profile.event_start_ns) {
                metrics->gpu_kernel_ms =
                    static_cast<double>(profile.event_end_ns - profile.event_start_ns) / 1e6;
            }
        } else if (!spec.sparse_on_npu) {
            metrics->gpu_kernel_ms = 0.0;
            std::fprintf(stderr,
                         "WARN: no OpenCL attention profile for line=%s; "
                         "sparse compute falls back to host timing\n", spec.name);
        }

        // GPU branch stages of the pair, now that the CL profile has fixed the
        // submit, wait and kernel numbers.
        int64_t sparse_kernel_end_us = 0;
        bool have_sparse_kernel = false;
        if (!spec.sparse_on_npu) {
            // The figure draws the sparse submit on the main path when the
            // calling thread does the enqueue, and on the GPU branch when the
            // sidecar submission thread owns it (route 3).
            const stage_lane submit_lane =
                spec.submit_thread ? stage_lane::gpu : stage_lane::main;
            const int64_t submit_span_end_us = submit_begin_us +
                static_cast<int64_t>(metrics->sparse_submit_ms * 1000.0);
            record_stage("submit sparse", stage_kind::host, submit_lane,
                         submit_begin_us, submit_span_end_us);
            // The kernel itself is always GPU work, whichever row its submit
            // came from.
            if (metrics->gpu_kernel_ms > 0.0) {
                if (has_profile && profile.event_end_ns > profile.event_start_ns &&
                    cl_event_on_host_clock(wait_end_us, profile.event_end_ns)) {
                    sparse_kernel_end_us = static_cast<int64_t>(profile.event_end_ns / 1000);
                    record_stage("sparse compute", stage_kind::gpu, stage_lane::gpu,
                                 static_cast<int64_t>(profile.event_start_ns / 1000),
                                 sparse_kernel_end_us);
                } else {
                    sparse_kernel_end_us =
                        submit_end_us + static_cast<int64_t>(metrics->gpu_kernel_ms * 1000.0);
                    record_stage("sparse compute", stage_kind::gpu, stage_lane::gpu,
                                 submit_end_us, sparse_kernel_end_us);
                }
                have_sparse_kernel = true;
            }
            record_stage("sync dense+sparse", stage_kind::host, stage_lane::main,
                         wait_begin_us, wait_end_us);
            // Route 3: the GPU branch is idle once its kernel is done, because
            // the residual it publishes has to wait for the NPU dense node.
            // Route 2 publishes no residual, so its idle tail stays grey
            // ("others").
            if (spec.residual_on_opencl && have_sparse_kernel &&
                sparse_kernel_end_us < dense_end_us) {
                record_stage("wait dense", stage_kind::waiting, stage_lane::gpu,
                             sparse_kernel_end_us, dense_end_us);
            }
        }

        // 4. Explicit copy out: the OpenCL result moves back into the NPU
        // allocation the residual and the NPU main path read.
        int64_t copy_out_us = 0;
        if (fx.separate_memory) {
            const int64_t begin = now_us();
            const bool moved = transfer_between_rpcmem(fx.sparse_result, fx.sparse);
            copy_out_us = now_us() - begin;
            if (!check(moved, "failed to copy the sparse result back into the HTP rpcmem")) {
                result = 2;
                break;
            }
            metrics->copy_out_bytes = static_cast<size_t>(ggml_nbytes(fx.sparse_result));
            record_stage("copy out", stage_kind::host, stage_lane::main,
                         begin, begin + copy_out_us);
        }
        metrics->copy_out_ms = static_cast<double>(copy_out_us) / 1000.0;

        // 5. Residual, on the main path, once the NPU branch has finished.
        // Route 3 queues the job and then blocks on it; the CPU-residual lines
        // compute it on the host.  The readbacks the test needs for its own
        // verification stay outside of the measured windows.
        int64_t pair_end_us = 0;
        if (spec.residual_on_opencl) {
            const int64_t queue_begin_us = now_us();
            if (!check_result(
                    ggml_backend_opencl_htp_graph_queue_exact_rho_residual_after_dense(
                        graph, fx.dense, reason, sizeof(reason)),
                    reason, "failed to queue the residual after the dense node")) {
                result = 1;
                break;
            }
            const int64_t queue_end_us = now_us();
            if (!check(wait_for_cache_publication(
                           sidecar, fx.residual_key, generation, /*timeout_ms=*/20000),
                       "residual job was not published")) {
                result = 1;
                break;
            }
            const int64_t publish_end_us = now_us();
            bool waited = false;
            if (!check_result(ggml_backend_opencl_htp_graph_wait_for_tensor(
                                  graph, fx.residual, &waited, reason, sizeof(reason)),
                              reason, "failed to wait for the OpenCL residual")) {
                result = 1;
                break;
            }
            const int64_t residual_end_us = now_us();
            pair_end_us = residual_end_us;
            // The queue call is main-path submission, the publication and the
            // completion wait are the main path blocking on the GPU branch, and
            // the exact-rho kernel itself runs inside that wait.
            record_stage("submit residual", stage_kind::host, stage_lane::main,
                         queue_begin_us, queue_end_us);
            // The main path blocks until the queued residual is published and
            // done, while the GPU branch runs the publication and the kernel.
            record_stage("sync residual", stage_kind::host, stage_lane::main,
                         queue_end_us, residual_end_us);
            record_stage("residual publish", stage_kind::host, stage_lane::gpu,
                         queue_end_us, publish_end_us);
            record_stage("residual compute", stage_kind::gpu, stage_lane::gpu,
                         publish_end_us, residual_end_us);

            ggml_backend_tensor_get(
                fx.dense, dense_out.data(), 0, dense_out.size() * sizeof(float));
            ggml_backend_tensor_get(
                sparse_consumer, sparse_out.data(), 0, sparse_out.size() * sizeof(float));
            compute_cpu_residual(test, dense_out, sparse_out, &expected_residual, &expected_rho);
            std::vector<ggml_fp16_t> residual_out(
                static_cast<size_t>(ggml_nelements(fx.residual)));
            ggml_backend_tensor_get(
                fx.residual, residual_out.data(), 0,
                residual_out.size() * sizeof(ggml_fp16_t));
            rho_out.assign(static_cast<size_t>(ggml_nelements(fx.rho)), 0.0f);
            ggml_backend_tensor_get(fx.rho, rho_out.data(), 0, rho_out.size() * sizeof(float));
            residual_max_abs = compare_fp16(residual_out, expected_residual);
        } else {
            // The CPU residual path reads the packed branches, computes rho and
            // the residual on the host and writes both back: for this line that
            // whole window is compute, not synchronization.
            const int64_t residual_begin_us = now_us();
            ggml_backend_tensor_get(
                fx.dense, dense_out.data(), 0, dense_out.size() * sizeof(float));
            ggml_backend_tensor_get(
                sparse_consumer, sparse_out.data(), 0, sparse_out.size() * sizeof(float));
            compute_cpu_residual(test, dense_out, sparse_out, &expected_residual, &expected_rho);
            ggml_backend_tensor_set(
                fx.residual, expected_residual.data(), 0,
                expected_residual.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(
                fx.rho, expected_rho.data(), 0, expected_rho.size() * sizeof(float));
            rho_out = expected_rho;
            const int64_t residual_end_us = now_us();
            pair_end_us = residual_end_us;
            metrics->residual_compute_ms =
                static_cast<double>(residual_end_us - residual_begin_us) / 1000.0;
            record_stage("cpu residual compute", stage_kind::host, stage_lane::main,
                         residual_begin_us, residual_end_us);
            // The CPU lines only produce the residual, so on them the checked
            // quantity is its magnitude; the OpenCL line compares its residual
            // against the very same host reference.
            double magnitude = 0.0;
            for (const ggml_fp16_t value : expected_residual) {
                magnitude = std::max(
                    magnitude,
                    std::fabs(static_cast<double>(ggml_fp16_to_fp32(value))));
            }
            residual_max_abs = magnitude;
        }

        metrics->residual_ms =
            static_cast<double>(pair_end_us - pair_begin_us) / 1000.0;
        // The stage spans tile the pair window; the leftover is the grey box.
        timeline.close(metrics->residual_ms);
        metrics->timeline = timeline;

        // Correctness gates.  They produce no reported number: the test
        // compares the branches and the residual against its own CPU
        // references and fails if they drift, which is what keeps the stage
        // numbers above meaningful.
        //
        // rho = exp(lse_sparse - lse_dense), so the kernel that applies it has
        // to reproduce what the two CPU references imply - a check that only
        // bites once the sparse branch really is a cheaper attention than the
        // dense one.
        double rho_error = 0.0;
        for (size_t row = 0; row < packed_rows(test); ++row) {
            const size_t lse_index = row * static_cast<size_t>(test.head_dim + 1) +
                static_cast<size_t>(test.head_dim);
            const double expected = std::exp(
                static_cast<double>(test.sparse_reference[lse_index]) -
                static_cast<double>(test.reference[lse_index]));
            const double measured = row < rho_out.size()
                ? static_cast<double>(rho_out[row])
                : expected;
            rho_error = std::max(rho_error, std::fabs(measured - expected));
        }
        const double dense_max_abs = compare_packed(dense_out, test.reference, test.head_dim);
        const double sparse_max_abs =
            compare_packed(sparse_out, test.sparse_reference, test.head_dim);

        if (!check(dense_max_abs <= k_tolerance,
                   "dense branch does not match the reference")) {
            result = 1;
        }
        if (!check(sparse_max_abs <= k_tolerance,
                   "sparse branch does not match the reference")) {
            result = 1;
        }
        if (!check(rho_error <= k_rho_tolerance,
                   "rho does not match exp(lse_sparse - lse_dense) of the references")) {
            result = 1;
        }
        // On the OpenCL line the residual is an error against the host
        // reference; the CPU lines only produce a magnitude, which must not be
        // zero (that would mean the sparse branch degenerated into the dense
        // one, which is what the all-kept mask does).
        if (spec.residual_on_opencl) {
            if (!check(residual_max_abs <= k_residual_tolerance,
                       "OpenCL residual does not match the host reference")) {
                result = 1;
            }
        } else if (!check(residual_max_abs > 0.0, "residual is identically zero")) {
            result = 1;
        }
    } while (false);

    // Teardown: finish and free the graph before the imported rpcmem buffers go
    // away, then drop the persistent imports (which also retires the residual
    // cache generations).
    if (graph != nullptr) {
        char finish_reason[k_reason_capacity] = {};
        const ggml_opencl_htp_result finish_result =
            ggml_backend_opencl_htp_graph_finish(graph, finish_reason, sizeof(finish_reason));
        if (result == 0 && finish_result != GGML_OPENCL_HTP_OK) {
            std::fprintf(stderr, "FAIL: sidecar graph finish: %s\n", finish_reason);
            result = 1;
        }
        ggml_backend_opencl_htp_graph_free(&graph);
    }
    if (residual_imported) {
        ggml_backend_opencl_htp_persistent_import_release(
            sidecar, fx.residual_key, generation, reason, sizeof(reason));
    }
    if (rho_imported) {
        ggml_backend_opencl_htp_persistent_import_release(
            sidecar, fx.rho_key, generation, reason, sizeof(reason));
    }
    free_line_fixture(fx);
    return result;
}

}  // namespace

int main() {
    attention_case test;
    test.image_tokens = env_int("HTP_OPENCL_BRANCH_IMAGE_TOKENS", 4096, k_block);
    test.text_tokens = env_int("HTP_OPENCL_BRANCH_TEXT_TOKENS", 154, 0);
    test.image_block_rows = env_int("HTP_OPENCL_BRANCH_IMAGE_BLOCK_ROWS", 16, 1);
    test.image_block_cols = env_int("HTP_OPENCL_BRANCH_IMAGE_BLOCK_COLS", 8, 1);
    if (test.image_tokens % k_block != 0) {
        std::fprintf(stderr, "FAIL: image token count must be a multiple of %d\n", k_block);
        return 2;
    }
    // The attention itself runs over the real joint sequence; only the QKV
    // storage is padded up to the next 32-token block boundary.
    test.tokens = test.image_tokens + test.text_tokens;
    test.tokens_alloc = ((test.tokens + k_block - 1) / k_block) * k_block;
    if (test.image_block_rows * test.image_block_cols * k_block != test.image_tokens) {
        std::fprintf(stderr,
                     "FAIL: image block grid %dx%d does not cover %d image tokens\n",
                     test.image_block_rows, test.image_block_cols, test.image_tokens);
        return 2;
    }
    test.heads = env_int("HTP_OPENCL_BRANCH_HEADS", 24, 1);
    test.kv_heads = env_int("HTP_OPENCL_BRANCH_KV_HEADS", 24, 1);
    test.batch = env_int("HTP_OPENCL_BRANCH_BATCH", 1, 1);
    if (test.heads % test.kv_heads != 0) {
        std::fprintf(stderr, "FAIL: query heads must be divisible by KV heads\n");
        return 2;
    }
    test.head_dim = env_int("HTP_OPENCL_BRANCH_HEAD_DIM", 64, 64);
    if (test.head_dim != 64 && test.head_dim != 128) {
        std::fprintf(stderr, "FAIL: head dim must be 64 or 128\n");
        return 2;
    }
    fill_case(test);

    const line_spec all_lines[] = {
        // npu-serial: route 1, both attentions on the NPU, one after the other,
        // residual on the CPU, no OpenCL work at all.
        {"npu-serial",      false, false, false, false, true,  "route1"},
        // parallel-copy: route 2, separate rpcmem, one plain async submit from
        // the calling thread, no submission queue, residual on CPU.
        {"parallel-copy",   true,  true,  false, false, false, "route2"},
        // parallel-shared: route 3, shared rpcmem, submission thread + early
        // submit, OpenCL residual.
        {"parallel-shared", false, true,  true,  true,  false, "route3"},
    };
    const char * requested = std::getenv("HTP_OPENCL_BRANCH_MODES");
    std::vector<line_spec> lines;
    if (requested == nullptr || requested[0] == '\0' || std::strcmp(requested, "all") == 0) {
        lines.assign(std::begin(all_lines), std::end(all_lines));
    } else {
        const std::string remaining = requested;
        size_t begin = 0;
        while (begin <= remaining.size()) {
            const size_t comma = remaining.find(',', begin);
            const std::string name = remaining.substr(
                begin, comma == std::string::npos ? std::string::npos : comma - begin);
            bool matched = false;
            for (const line_spec & spec : all_lines) {
                if (name == spec.name) {
                    lines.push_back(spec);
                    matched = true;
                    break;
                }
            }
            if (!matched && !name.empty()) {
                std::fprintf(stderr, "FAIL: unknown line %s\n", name.c_str());
                return 2;
            }
            if (comma == std::string::npos) {
                break;
            }
            begin = comma + 1;
        }
    }
    if (lines.empty()) {
        std::fprintf(stderr, "FAIL: no line selected\n");
        return 2;
    }

    // The HTP runtime and the sidecar both read their flags at construction
    // time, so every baseline flag has to be in place before either exists.
    configure_common_route();
    set_env("SD_HTP_SPARSE_OPENCL_RESIDUAL", "0");

    ggml_backend_htp_set_hybrid_enabled(true);
    ggml_backend_t backend = ggml_backend_init_by_name("HTP", nullptr);
    if (backend == nullptr || !ggml_backend_htp_runtime_available()) {
        std::fprintf(stderr, "SKIP: HTP runtime unavailable for the branch matrix: %s\n",
                     ggml_backend_htp_runtime_error() != nullptr
                         ? ggml_backend_htp_runtime_error() : "unknown");
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
        ggml_backend_htp_set_hybrid_enabled(false);
        return k_skip;
    }
    ggml_backend_htp_set_n_threads(backend, 8);

    char reason[k_reason_capacity] = {};
    ggml_opencl_qcom_requirements requirements = {};
    ggml_opencl_htp_sidecar * sidecar =
        ggml_backend_opencl_htp_sidecar_create(&requirements, reason, sizeof(reason));
    if (sidecar == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL sidecar unavailable: %s\n",
                     reason[0] != '\0' ? reason : "unknown");
        ggml_backend_free(backend);
        ggml_backend_htp_set_hybrid_enabled(false);
        return k_skip;
    }
    std::printf(
        "sidecar platform=%s device=%s queue_in_order=%d image_tokens=%d text_tokens=%d "
        "real_tokens=%d tokens=%d tokens_alloc=%d blocks=%d grid=%dx%d heads=%d kv_heads=%d "
        "head_dim=%d batch=%d\n",
        ggml_backend_opencl_htp_sidecar_platform_vendor(sidecar),
        ggml_backend_opencl_htp_sidecar_device_name(sidecar),
        ggml_backend_opencl_htp_sidecar_queue_is_in_order(sidecar) ? 1 : 0,
        test.image_tokens, test.text_tokens, real_tokens(test), test.tokens, test.tokens_alloc,
        q_blocks(test), test.image_block_rows, test.image_block_cols,
        test.heads, test.kv_heads, test.head_dim, test.batch);
    std::fflush(stdout);

    uint64_t generation = 1;
    int result = 0;
    // Compile the OpenCL kernels and let the driver warm its program cache once
    // per line before anything is measured.
    for (const line_spec & spec : lines) {
        set_env("SD_HTP_SPARSE_OPENCL_RESIDUAL", spec.residual_on_opencl ? "1" : "0");
        line_metrics scratch;
        const int warm = run_line(backend, sidecar, test, spec, generation++, &scratch);
        if (warm != 0) {
            result = warm;
            break;
        }
    }

    const int repeat = env_int("HTP_OPENCL_BRANCH_REPEAT", 1, 1);
    std::vector<std::vector<line_metrics>> samples(lines.size());
    for (int iteration = 0; iteration < repeat && result == 0; ++iteration) {
        for (size_t i = 0; i < lines.size(); ++i) {
            set_env("SD_HTP_SPARSE_OPENCL_RESIDUAL",
                    lines[i].residual_on_opencl ? "1" : "0");
            line_metrics metrics;
            const int line_result =
                run_line(backend, sidecar, test, lines[i], generation++, &metrics);
            if (line_result != 0) {
                result = line_result;
                break;
            }
            // Every raw observation is printed so the spread stays visible;
            // the summary below reports the mean of each column.
            print_line(lines[i].name, metrics, "run");
            samples[i].push_back(metrics);
        }
    }

    if (result == 0) {
        for (size_t i = 0; i < lines.size(); ++i) {
            print_line(lines[i].name, summarise_line(samples[i]), "mean");
        }
        std::printf(
            "legend: one `stage` row per drawn box of the pair timeline "
            "(line, tag, route, lane, name, kind, begin_ms, end_ms, dur_ms, share); "
            "kind=host submit/sync/notify, npu compute, gpu compute, waiting, others; "
            "lane=main is the path the caller drives, lane=gpu is the sparse/residual "
            "branch; share is the fraction of the pair.  `stage_totals` sums the kinds "
            "per lane, `timeline` draws one column per percent of the pair with "
            "H=host, N=npu, G=gpu, W=waiting and a space for the rest "
            "(HTP_OPENCL_BRANCH_TIMELINE=0 drops the bars).\n");
        std::printf(
            "copy: copy_in/out_mb are the bytes moved between the two rpcmem "
            "allocations and back, copy_in/out_ms the time those transfers took on the "
            "main path; the lines that share one allocation report zeros.\n");
        std::printf(
            "routes: route1 NPU dense + NPU sparse serial with a CPU residual; route2 "
            "NPU dense + GPU sparse concurrent with the explicit copies and a CPU "
            "residual; route3 shared rpcmem, early submit, OpenCL residual.\n");
        std::puts("PASS htp-opencl-branch-modes");
        std::fflush(stdout);
    }

    const size_t live_imports =
        ggml_backend_opencl_htp_sidecar_live_persistent_import_count(sidecar);
    const size_t live_events =
        ggml_backend_opencl_htp_sidecar_live_cache_event_count(sidecar);
    if (result == 0) {
        if (!check(live_imports == 0, "persistent imports leaked across the lines") ||
            !check(live_events == 0, "residual cache events leaked across the lines")) {
            result = 1;
        }
    }

    ggml_backend_opencl_htp_sidecar_free(sidecar);
    ggml_backend_free(backend);
    ggml_backend_htp_set_hybrid_enabled(false);
    unset_env("SD_HTP_SPARSE_OPENCL");
    unset_env("SD_HTP_SPARSE_OPENCL_PROFILE");
    unset_env("SD_HTP_SPARSE_OPENCL_RESIDUAL");
    return result;
}
