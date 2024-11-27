#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

#define PARAMS(T) \
    device T* __q__ [[buffer(0)]], \
    device T* __k__ [[buffer(1)]], \
    device T* __v__ [[buffer(2)]], \
    device T* __o__ [[buffer(3)]], \
    constant unsigned &H [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint  laneId   [[thread_index_in_simdgroup]] \

constant constexpr const int TN = 8;
template <int D>
kernel void attn_fwd(device bf16* q [[buffer(0)]], device bf16* k [[buffer(1)]], device bf16* v [[buffer(2)]], device bf16* o [[buffer(3)]],
                       constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]],
                       uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = mittens::gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_k(k, nullptr, H, N, nullptr);
    global_layout gl_v(v, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    
    rt_bf<8, D> q_reg, v_reg;
    rt_bf<8, D, ducks::rt_layout::col> k_reg;
    rt_fl<8, 8> att_block;
    rt_fl<8, D> o_reg;
    rt_fl<8, 8>::col_vec max_vec, max_vec_last, norm_vec;
    
    const int block = blockIdx.z, head = blockIdx.y;
    const int q_seq = blockIdx.x;
    const int kv_blocks = N / v_reg.rows;
    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);
    constexpr const bf16 q_mul = ((D == 128) ? 0.08838834764bf : 0.125bf) * 1.44269504089bf;
    mul(q_reg, q_reg, q_mul);
    #pragma clang loop unroll(full)
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        copy(max_vec_last,  max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        //        subexp2(max_vec_last, max_vec_last, max_vec);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        //        subexp2(att_block, att_block, max_vec);
        
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, gl_v, {block, head, kv_idx, 0}, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}

#define instantiate_add_custom(D)                                \
  template [[host_name("attn_fwd_" #D)]] [[kernel]] void         \
  attn_fwd<D>(device bf16* q [[buffer(0)]], device bf16* k [[buffer(1)]], device bf16* v [[buffer(2)]], device bf16* o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);

instantiate_add_custom(64);
instantiate_add_custom(128);

}
