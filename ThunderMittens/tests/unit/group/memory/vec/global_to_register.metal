#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER

#include "../../../../../include/tk.metal"
namespace mittens {

#define gtr_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3          id              [[thread_position_in_grid]], \
    uint          threadIdx       [[thread_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]
//constant constexpr int B = 1, D = 1, R = 4, C = 5;
//constant constexpr int B = 1, D = 1, R = 1, C = 2;
template<typename T, unsigned L, unsigned NW, typename layout>
kernel void group_vec_gtr(gtr_kern_PARAMS(T))
{
    using GL = gl<T, 1, 1, 1, 8*L * NW>;
    using reg_vec = rv<T, L * 8, layout>;
    using load_group = group<NW>;
    reg_vec vec1;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load_group::load(vec1, input, {}, threadIdx);
    
    load_group::store(output, vec1, {}, threadIdx);
}

#define gen_kernel_NW(reg_N, NW) \
    template [[host_name("group_vec_gtr_loadstore_half_" #reg_N "_" #NW "warps_rv_align_layout")]] [[kernel]] \
    void group_vec_gtr<half, reg_N, NW, ducks::rv_layout::align>(gtr_kern_PARAMS(half)); \
\
    template [[host_name("group_vec_gtr_loadstore_half_" #reg_N "_" #NW "warps_rv_ortho_layout")]] [[kernel]] \
    void group_vec_gtr<half, reg_N, NW, ducks::rv_layout::ortho>(gtr_kern_PARAMS(half)); \
\
    template [[host_name("group_vec_gtr_loadstore_half_" #reg_N "_" #NW "warps_rv_naive_layout")]] [[kernel]] \
    void group_vec_gtr<half, reg_N, NW, ducks::rv_layout::naive>(gtr_kern_PARAMS(half)); \
\
\
    template [[host_name("group_vec_gtr_loadstore_float_" #reg_N "_" #NW "warps_rv_align_layout")]] [[kernel]] \
    void group_vec_gtr<float, reg_N, NW, ducks::rv_layout::align>(gtr_kern_PARAMS(float)); \
\
    template [[host_name("group_vec_gtr_loadstore_float_" #reg_N "_" #NW "warps_rv_ortho_layout")]] [[kernel]] \
    void group_vec_gtr<float, reg_N, NW, ducks::rv_layout::ortho>(gtr_kern_PARAMS(float)); \
\
    template [[host_name("group_vec_gtr_loadstore_float_" #reg_N "_" #NW "warps_rv_naive_layout")]] [[kernel]] \
    void group_vec_gtr<float, reg_N, NW, ducks::rv_layout::naive>(gtr_kern_PARAMS(float)); \
\
\
    template [[host_name("group_vec_gtr_loadstore_bf16_" #reg_N "_" #NW "warps_rv_align_layout")]] [[kernel]] \
    void group_vec_gtr<bf16, reg_N, NW, ducks::rv_layout::align>(gtr_kern_PARAMS(bf16)); \
\
    template [[host_name("group_vec_gtr_loadstore_bf16_" #reg_N "_" #NW "warps_rv_ortho_layout")]] [[kernel]] \
    void group_vec_gtr<bf16, reg_N, NW, ducks::rv_layout::ortho>(gtr_kern_PARAMS(bf16)); \
\
    template [[host_name("group_vec_gtr_loadstore_bf16_" #reg_N "_" #NW "warps_rv_naive_layout")]] [[kernel]] \
    void group_vec_gtr<bf16, reg_N, NW, ducks::rv_layout::naive>(gtr_kern_PARAMS(bf16)); \



#define gen_kernels(reg_N) \
    gen_kernel_NW(reg_N, 2) gen_kernel_NW(reg_N, 3) gen_kernel_NW(reg_N, 4)

#if (TEST_INTENSITY==1)

    gen_kernels(1) gen_kernels(2)
#elif (TEST_INTENSITY==2)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
#elif (TEST_INTENSITY==3) // 8
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
#elif (TEST_INTENSITY==4) // 16
    gen_kernels(1) gen_kernels(2)
    gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
#endif

#undef gen_kernel
    
}


#endif
