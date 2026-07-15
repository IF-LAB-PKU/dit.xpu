#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HEAD_DIM 64
#define BLOCK_SIZE 32
#define VEC_COUNT 16
#define MASK_FORMAT_BLOCK_INDEX 2

__attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
__kernel void flash_attn_block_sparse_f32_f16(
        __global const uchar * q_base,
        ulong q_offset,
        __global const uchar * k_base,
        ulong k_offset,
        __global const uchar * v_base,
        ulong v_offset,
        __global uchar * o_base,
        ulong o_offset,
        __global const uchar * mask_base,
        ulong mask_offset,
        float scale,
        int Lq,
        int Lk,
        int H,
        int Hkv,
        int mask_format,
        int max_kept_blocks,
        ulong q_nb1,
        ulong q_nb2,
        ulong q_nb3,
        ulong k_nb1,
        ulong k_nb2,
        ulong k_nb3,
        ulong v_nb1,
        ulong v_nb2,
        ulong v_nb3,
        ulong o_nb1,
        ulong o_nb2,
        ulong o_nb3,
        ulong mask_nb0,
        ulong mask_nb1) {
    const int lane = (int)get_local_id(0);
    const int query_block = (int)get_group_id(0);
    const int head_batch = (int)get_global_id(1);
    const int h = head_batch % H;
    const int b = head_batch / H;
    const int qt = query_block * BLOCK_SIZE + lane;
    const int active_query = qt < Lq;
    const int q_blocks_per_head = 1 + (Lq - 1) / BLOCK_SIZE;
    const int k_blocks_per_head = 1 + (Lk - 1) / BLOCK_SIZE;
    const int heads_per_kv = H / Hkv;
    const int kvh = h / heads_per_kv;
    const int qb_abs = h * q_blocks_per_head + query_block;

    float4 q[VEC_COUNT];
    float4 output[VEC_COUNT];
    for (int i = 0; i < VEC_COUNT; ++i) {
        q[i] = (float4)(0.0f);
        output[i] = (float4)(0.0f);
    }
    if (active_query) {
        __global const uchar * q_row =
            q_base + q_offset + (ulong)qt * q_nb1 + (ulong)h * q_nb2 + (ulong)b * q_nb3;
        for (int i = 0; i < VEC_COUNT; ++i) {
            q[i] = vload4(0, (__global const float *)(q_row + (ulong)(i * 4 * sizeof(float))));
        }
    }

    __local half4 k_tile[BLOCK_SIZE][VEC_COUNT];
    __local half4 v_tile[BLOCK_SIZE][VEC_COUNT];
    float maximum = -INFINITY;
    float sum = 0.0f;

    const int candidate_count =
        mask_format == MASK_FORMAT_BLOCK_INDEX ? max_kept_blocks : k_blocks_per_head;
    // Production index rows contain unique valid block ids; duplicates are outside this format contract.
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        int kb_local = candidate;
        if (mask_format == MASK_FORMAT_BLOCK_INDEX) {
            const int kb_abs = *(__global const int *)(
                mask_base + mask_offset + (ulong)candidate * mask_nb0 + (ulong)qb_abs * mask_nb1);
            if (kb_abs == -1) {
                break;
            }
            if (kb_abs < 0 || kb_abs >= Hkv * k_blocks_per_head ||
                    kb_abs / k_blocks_per_head != kvh) {
                continue;
            }
            kb_local = kb_abs - kvh * k_blocks_per_head;
        } else {
            const int kb_abs = kvh * k_blocks_per_head + kb_local;
            const int mask_value = *(__global const int *)(
                mask_base + mask_offset + (ulong)kb_abs * mask_nb0 + (ulong)qb_abs * mask_nb1);
            if (mask_value != 0) {
                continue;
            }
        }

        const int key_start = kb_local * BLOCK_SIZE;
        const int valid_keys = min(BLOCK_SIZE, Lk - key_start);
        const int kt = key_start + lane;
        if (lane < valid_keys) {
            __global const uchar * k_row =
                k_base + k_offset + (ulong)kt * k_nb1 + (ulong)kvh * k_nb2 + (ulong)b * k_nb3;
            __global const uchar * v_row =
                v_base + v_offset + (ulong)kt * v_nb1 + (ulong)kvh * v_nb2 + (ulong)b * v_nb3;
            for (int i = 0; i < VEC_COUNT; ++i) {
                k_tile[lane][i] = vload4(
                    0, (__global const half *)(k_row + (ulong)(i * 4 * sizeof(half))));
                v_tile[lane][i] = vload4(
                    0, (__global const half *)(v_row + (ulong)(i * 4 * sizeof(half))));
            }
        } else {
            for (int i = 0; i < VEC_COUNT; ++i) {
                k_tile[lane][i] = (half4)(0.0f);
                v_tile[lane][i] = (half4)(0.0f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active_query) {
            for (int j = 0; j < valid_keys; ++j) {
                float score = 0.0f;
                for (int i = 0; i < VEC_COUNT; ++i) {
                    score += dot(q[i], convert_float4(k_tile[j][i]));
                }
                score *= scale;

                const float next_maximum = fmax(maximum, score);
                const float previous_scale = sum == 0.0f ? 0.0f : exp(maximum - next_maximum);
                const float weight = exp(score - next_maximum);
                for (int i = 0; i < VEC_COUNT; ++i) {
                    output[i] = output[i] * previous_scale + convert_float4(v_tile[j][i]) * weight;
                }
                sum = sum * previous_scale + weight;
                maximum = next_maximum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!active_query) {
        return;
    }

    __global float * output_row = (__global float *)(
        o_base + o_offset + (ulong)h * o_nb1 + (ulong)qt * o_nb2 + (ulong)b * o_nb3);
    if (sum == 0.0f) {
        for (int i = 0; i < VEC_COUNT; ++i) {
            vstore4((float4)(0.0f), 0, output_row + i * 4);
        }
        output_row[HEAD_DIM] = -INFINITY;
        return;
    }

    const float inverse_sum = 1.0f / sum;
    for (int i = 0; i < VEC_COUNT; ++i) {
        vstore4(output[i] * inverse_sum, 0, output_row + i * 4);
    }
    output_row[HEAD_DIM] = maximum + log(sum);
}
