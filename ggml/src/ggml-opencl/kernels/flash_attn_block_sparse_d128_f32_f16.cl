#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable

// D128 分块稀疏 Flash Attention kernel。
// pair64 是 host 为满足条件的 D128 BLOCK_INDEX 工作负载选择的正式路径。
// 一个 WG64 处理包含 32 个 query 的分块，每个 query 由两个 lane 协作，
// 每个 lane 负责 64 个 head 维度。
//
// K 和 V 每次以一个包含 32 个 key 的分块暂存到 local memory，Q 和未归一化的 output 累加器保存在 private memory
// 每个输出行先存放 128 个归一化结果
// 再在 output[HEAD_DIM] 存放 log-sum-exp

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
// Qualcomm 的 "half" 对应一个 64-lane subgroup。
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#define HEAD_DIM 128
#define BLOCK_M 32
#define BLOCK_N 32
#define MASK_BLOCK_SIZE 32
#define VEC_COUNT 32
#define PAIR_VEC_COUNT 16
#define MASK_FORMAT_BLOCK_INDEX 2

REQD_SUBGROUP_SIZE_64
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void flash_attn_block_sparse_d128_pair64_f32_f16(
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
        ulong v_token_stride,
        ulong v_head_stride,
        ulong v_batch_stride,
        ulong o_nb1,
        ulong o_nb2,
        ulong o_nb3,
        ulong mask_nb0,
        ulong mask_nb1) {
    // lane (2q, 2q+1) 协作处理 query q。dimension_lane 选择前 64 或后 64 个维度，每一半由 16 个 float4 表示。
    const int lane = (int) get_local_id(0);
    const int query_lane = lane >> 1;
    const int dimension_lane = lane & 1;
    const int vector_start = dimension_lane * PAIR_VEC_COUNT;
    const int query_block = (int) get_group_id(0);
    const int head_batch = (int) get_global_id(1);
    const int h = head_batch % H;
    const int b = head_batch / H;
    const int qt = query_block * BLOCK_M + query_lane;
    const int active_query = qt < Lq;
    const int q_blocks_per_head = 1 + (Lq - 1) / MASK_BLOCK_SIZE;
    const int k_blocks_per_head = 1 + (Lk - 1) / MASK_BLOCK_SIZE;
    const int heads_per_kv = H / Hkv;
    const int kvh = h / heads_per_kv;
    // block-index mask 按 query head 排列，其中的值用于寻址各 KV head 的 key block。
    const int qb_abs = h * q_blocks_per_head + query_block;

    // 将 D128 拆分到两个 lane，使每个 lane 的 Q/output private 状态减半。
    float4 q[PAIR_VEC_COUNT];
    float4 output[PAIR_VEC_COUNT];
#pragma unroll
    for (int i = 0; i < PAIR_VEC_COUNT; ++i) {
        q[i] = (float4) (0.0f);
        output[i] = (float4) (0.0f);
    }
    if (active_query) {
        __global const uchar * q_row =
            q_base + q_offset + (ulong) qt * q_nb1 + (ulong) h * q_nb2 + (ulong) b * q_nb3;
#pragma unroll
        for (int i = 0; i < PAIR_VEC_COUNT; ++i) {
            q[i] = vload4(
                0,
                (__global const float *)
                    (q_row + (ulong) ((vector_start + i) * 4 * sizeof(float))));
        }
    }

    __local half4 k_tile[BLOCK_N][VEC_COUNT];
    __local half4 v_tile[BLOCK_N][VEC_COUNT];
    float maximum = -INFINITY;
    float sum = 0.0f;

    if (mask_format != MASK_FORMAT_BLOCK_INDEX) {
        return;
    }

    // 每个候选 block id 只由 subgroup 的 lane 0 读取，再广播给其他 lane，
    // 避免 64 次相同的 global load，并让所有 lane 保持一致的控制流。
    for (int candidate = 0; candidate < max_kept_blocks; ++candidate) {
        int kb_abs = -1;
        if (get_sub_group_local_id() == 0) {
            kb_abs = *(__global const int *)
                (mask_base + mask_offset + (ulong) candidate * mask_nb0 +
                 (ulong) qb_abs * mask_nb1);
        }
        kb_abs = sub_group_broadcast(kb_abs, 0);
        if (kb_abs == -1) {
            break;
        }
        if (kb_abs < 0 || kb_abs >= Hkv * k_blocks_per_head ||
            kb_abs / k_blocks_per_head != kvh) {
            continue;
        }

        const int kb_local = kb_abs - kvh * k_blocks_per_head;
        const int key_start = kb_local * MASK_BLOCK_SIZE;
        const int valid_keys = min(MASK_BLOCK_SIZE, Lk - key_start);
        // 协作暂存一个 [32 keys, 128 dimensions] 的 K/V tile。超过 Lk 的行补零，
        // 确保向量读取不会越界。
        for (int flat = lane; flat < BLOCK_N * VEC_COUNT; flat += 64) {
            const int row = flat / VEC_COUNT;
            const int column = flat % VEC_COUNT;
            const int kt = key_start + row;
            if (row < valid_keys) {
                __global const uchar * k_row =
                    k_base + k_offset + (ulong) kt * k_nb1 +
                    (ulong) kvh * k_nb2 + (ulong) b * k_nb3;
                __global const uchar * v_row =
                    v_base + v_offset + (ulong) kt * v_token_stride +
                    (ulong) kvh * v_head_stride + (ulong) b * v_batch_stride;
                k_tile[row][column] = vload4(
                    0,
                    (__global const half *)
                        (k_row + (ulong) (column * 4 * sizeof(half))));
                v_tile[row][column] = vload4(
                    0,
                    (__global const half *)
                        (v_row + (ulong) (column * 4 * sizeof(half))));
            } else {
                k_tile[row][column] = (half4) (0.0f);
                v_tile[row][column] = (half4) (0.0f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active_query) {
            // 每次 online softmax 更新处理两个 key。每个 lane 计算 64 维的部分点积，
            // xor(1) 将属于同一 query 的两个 lane 合并。
            for (int j = 0; j < valid_keys; j += 2) {
                float4 dot_acc0 = (float4) (0.0f);
                float4 dot_acc1 = (float4) (0.0f);
#pragma unroll
                for (int i = 0; i < PAIR_VEC_COUNT; ++i) {
                    const int vector_index = vector_start + i;
                    dot_acc0 = mad(
                        q[i], convert_float4(k_tile[j][vector_index]), dot_acc0);
                    dot_acc1 = mad(
                        q[i], convert_float4(k_tile[j + 1][vector_index]), dot_acc1);
                }
                float partial0 = dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3;
                float partial1 = dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3;
                partial0 += sub_group_shuffle_xor(partial0, 1);
                partial1 += sub_group_shuffle_xor(partial1, 1);
                const float score0 = partial0 * scale;
                const float score1 = j + 1 < valid_keys ? partial1 * scale : -INFINITY;

                // 数值稳定的 online softmax：当运行中的最大值增大时，重新缩放已累积的
                // 分子和分母。
                const float next_maximum = fmax(maximum, fmax(score0, score1));
                const float previous_scale =
                    sum == 0.0f ? 0.0f : exp(maximum - next_maximum);
                const float weight0 = exp(score0 - next_maximum);
                const float weight1 = exp(score1 - next_maximum);
#pragma unroll
                for (int i = 0; i < PAIR_VEC_COUNT; ++i) {
                    const int vector_index = vector_start + i;
                    output[i] = output[i] * previous_scale +
                        convert_float4(v_tile[j][vector_index]) * weight0 +
                        convert_float4(v_tile[j + 1][vector_index]) * weight1;
                }
                sum = sum * previous_scale + weight0 + weight1;
                maximum = next_maximum;
            }
        }
        // 在所有有效 query 使用完当前 key block 前，任何 lane 都不能覆盖 local K/V。
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!active_query) {
        return;
    }

    __global float * output_row = (__global float *)
        (o_base + o_offset + (ulong) h * o_nb1 + (ulong) qt * o_nb2 +
         (ulong) b * o_nb3);
    // 空稀疏行输出零向量，并令 LSE=-inf。
    if (sum == 0.0f) {
#pragma unroll
        for (int i = 0; i < PAIR_VEC_COUNT; ++i) {
            vstore4((float4) (0.0f), 0, output_row + (vector_start + i) * 4);
        }
        if (dimension_lane == 0) {
            output_row[HEAD_DIM] = -INFINITY;
        }
        return;
    }

    const float inverse_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < PAIR_VEC_COUNT; ++i) {
        vstore4(output[i] * inverse_sum, 0, output_row + (vector_start + i) * 4);
    }
    // 两个 dimension lane 的 softmax 状态相同，只由其中一个 lane 写入 LSE。
    if (dimension_lane == 0) {
        output_row[HEAD_DIM] = maximum + log(sum);
    }
}
