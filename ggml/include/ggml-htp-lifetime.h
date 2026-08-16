#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ggml_htp_opencl_lifetime_mode {
    GGML_HTP_OPENCL_LIFETIME_INVALID = -1,
    GGML_HTP_OPENCL_LIFETIME_DEFERRED = 0,
    GGML_HTP_OPENCL_LIFETIME_REUSE = 1,
    GGML_HTP_OPENCL_LIFETIME_STRICT = 2,
} ggml_htp_opencl_lifetime_mode;

static inline ggml_htp_opencl_lifetime_mode
ggml_htp_opencl_lifetime_mode_parse(const char * value) {
    if (value == NULL) {
        return GGML_HTP_OPENCL_LIFETIME_DEFERRED;
    }
    if (strcmp(value, "deferred") == 0) {
        return GGML_HTP_OPENCL_LIFETIME_DEFERRED;
    }
    if (strcmp(value, "reuse") == 0) {
        return GGML_HTP_OPENCL_LIFETIME_REUSE;
    }
    if (strcmp(value, "strict") == 0) {
        return GGML_HTP_OPENCL_LIFETIME_STRICT;
    }
    return GGML_HTP_OPENCL_LIFETIME_INVALID;
}

static inline const char *
ggml_htp_opencl_lifetime_mode_name(ggml_htp_opencl_lifetime_mode mode) {
    switch (mode) {
        case GGML_HTP_OPENCL_LIFETIME_DEFERRED:
            return "deferred";
        case GGML_HTP_OPENCL_LIFETIME_REUSE:
            return "reuse";
        case GGML_HTP_OPENCL_LIFETIME_STRICT:
            return "strict";
        case GGML_HTP_OPENCL_LIFETIME_INVALID:
            break;
    }
    return "invalid";
}

static inline bool ggml_htp_opencl_lifetime_uses_global_mask_sweep(
        ggml_htp_opencl_lifetime_mode mode) {
    return mode == GGML_HTP_OPENCL_LIFETIME_DEFERRED;
}

static inline bool
ggml_htp_opencl_staging_slots_parse(const char * value, int * slots) {
    if (slots == NULL) {
        return false;
    }
    if (value == NULL) {
        *slots = 2;
        return true;
    }
    if (value[0] < '1' || value[0] > '8' || value[1] != '\0') {
        return false;
    }
    *slots = value[0] - '0';
    return true;
}

typedef struct ggml_htp_opencl_pair_lifetime_state {
    bool sparse_committed;
    bool sparse_complete;
    bool residual_submitted;
    bool residual_complete;
    bool reaped;
} ggml_htp_opencl_pair_lifetime_state;

static inline bool ggml_htp_opencl_pair_mark_sparse_committed(
        ggml_htp_opencl_pair_lifetime_state * state) {
    if (state == NULL || state->reaped) {
        return false;
    }
    state->sparse_committed = true;
    return true;
}

static inline bool ggml_htp_opencl_pair_mark_sparse_complete(
        ggml_htp_opencl_pair_lifetime_state * state) {
    if (state == NULL || state->reaped || !state->sparse_committed) {
        return false;
    }
    state->sparse_complete = true;
    return true;
}

static inline bool ggml_htp_opencl_pair_mark_residual_submitted(
        ggml_htp_opencl_pair_lifetime_state * state) {
    if (state == NULL || state->reaped || !state->sparse_committed ||
        state->residual_complete) {
        return false;
    }
    state->residual_submitted = true;
    return true;
}

static inline bool ggml_htp_opencl_pair_mark_residual_complete(
        ggml_htp_opencl_pair_lifetime_state * state) {
    if (state == NULL || state->reaped || !state->residual_submitted) {
        return false;
    }
    state->residual_complete = true;
    return true;
}

static inline bool ggml_htp_opencl_pair_mark_reaped(
        ggml_htp_opencl_pair_lifetime_state * state) {
    if (state == NULL || !state->sparse_complete ||
        !state->residual_complete) {
        return false;
    }
    state->reaped = true;
    return true;
}

static inline bool
ggml_htp_opencl_pair_reap_requires_residual_submission_wait(
        const ggml_htp_opencl_pair_lifetime_state * state,
        bool wait_for_completion) {
    return state != NULL && wait_for_completion && !state->reaped &&
        state->sparse_complete && !state->residual_submitted;
}

static inline int ggml_htp_opencl_staging_pool_select_slot(
        const size_t * capacities,
        const bool * in_use,
        size_t slot_count,
        size_t slot_limit,
        size_t required_bytes,
        bool * resize) {
    if (capacities == NULL || in_use == NULL || resize == NULL ||
        slot_limit == 0 || required_bytes == 0 || slot_count > slot_limit) {
        return -1;
    }

    size_t best_capacity = (size_t) -1;
    int best = -1;
    for (size_t i = 0; i < slot_count; ++i) {
        if (!in_use[i] && capacities[i] >= required_bytes &&
            capacities[i] < best_capacity) {
            best = (int) i;
            best_capacity = capacities[i];
        }
    }
    if (best >= 0) {
        *resize = false;
        return best;
    }

    for (size_t i = 0; i < slot_count; ++i) {
        if (!in_use[i]) {
            *resize = true;
            return (int) i;
        }
    }
    if (slot_count < slot_limit) {
        *resize = true;
        return (int) slot_count;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif
