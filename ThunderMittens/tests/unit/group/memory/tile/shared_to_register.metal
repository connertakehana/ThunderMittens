#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER

#include "../../../../../include/tk.metal"
namespace mittens {

#define gtr_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3          id              [[thread_position_in_grid]], \
    uint          threadIdx       [[thread_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]

template<typename T, unsigned H, unsigned W, unsigned NW, typename L>
kernel void group_str(gtr_kern_PARAMS(T))
{
    const int B = 3, D = 1, R = NW, C = 5;
    using shared_tile = st<T, H * 8 * NW, W * 8>;
    using reg_tile    = rt<T, H * 8,     W * 8, L>;
    using GL = gl<T, B, D, 8*R*H, 8*C*W>; // 32 * 4 = 128
    using load_group = group<NW>;
    threadgroup shared_tile smem_A;
    threadgroup shared_tile smem_B;
    reg_tile reg_A;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    for(int i = 0; i < input.batch; i++)
    for(int j = 0; j < input.depth; j++)
    for(int k = 0; k < input.rows/shared_tile::rows; k++)
    for(int l = 0; l < input.cols/shared_tile::cols; l++) {
        load_group::load(smem_A, input, {i, j, k, l}, threadIdx);
        metal::threadgroup_barrier(metal::mem_flags::mem_none);
        load_group::load(reg_A, smem_A, threadIdx);
        load_group::store(smem_B, reg_A, threadIdx);
        metal::threadgroup_barrier(metal::mem_flags::mem_none);
        load_group::store(output, smem_B, {i, j, k, l}, threadIdx);
        
    }
}

#define gen_kernel_NW(reg_N, reg_M, NW) \
    template [[host_name("group_str_loadstore_half_" #reg_N "x" #reg_M "_" #NW "warps_rt_row_layout")]] [[kernel]] \
    void group_str<half, reg_N, reg_M, NW, ducks::rt_layout::row>(gtr_kern_PARAMS(half)); \
\
    template [[host_name("group_str_loadstore_half_" #reg_N "x" #reg_M "_" #NW "warps_rt_col_layout")]] [[kernel]] \
    void group_str<half, reg_N, reg_M, NW, ducks::rt_layout::col>(gtr_kern_PARAMS(half)); \
\
    template [[host_name("group_str_loadstore_float_" #reg_N "x" #reg_M "_" #NW "warps_rt_row_layout")]] [[kernel]] \
    void group_str<float, reg_N, reg_M, NW, ducks::rt_layout::row>(gtr_kern_PARAMS(float)); \
\
    template [[host_name("group_str_loadstore_float_" #reg_N "x" #reg_M "_" #NW "warps_rt_col_layout")]] [[kernel]] \
    void group_str<float, reg_N, reg_M, NW, ducks::rt_layout::col>(gtr_kern_PARAMS(float)); \
\
    template [[host_name("group_str_loadstore_bf16_" #reg_N "x" #reg_M "_" #NW "warps_rt_row_layout")]] [[kernel]] \
    void group_str<bf16, reg_N, reg_M, NW, ducks::rt_layout::row>(gtr_kern_PARAMS(bf16)); \
\
    template [[host_name("group_str_loadstore_bf16_" #reg_N "x" #reg_M "_" #NW "warps_rt_col_layout")]] [[kernel]] \
    void group_str<bf16, reg_N, reg_M, NW, ducks::rt_layout::col>(gtr_kern_PARAMS(bf16));


#define gen_kernel(reg_N, reg_M) \
    gen_kernel_NW(reg_N, reg_M, 2) gen_kernel_NW(reg_N, reg_M, 3) gen_kernel_NW(reg_N, reg_M, 4)

#if (TEST_INTENSITY==1)
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)

    gen_kernels(1) gen_kernels(2)
    #undef gen_kernels
#elif (TEST_INTENSITY==2)
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)  gen_kernel(N,3)  gen_kernel(N,4)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
    #undef gen_kernels
#elif (TEST_INTENSITY==3) // 8
    
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)  gen_kernel(N,3)  gen_kernel(N,4)  gen_kernel(N,5)  gen_kernel(N,6)  gen_kernel(N,7)  gen_kernel(N,8)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    #undef gen_kernels
#elif (TEST_INTENSITY==4) // 16
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)  gen_kernel(N,3)  gen_kernel(N,4)  gen_kernel(N,5)  gen_kernel(N,6)  gen_kernel(N,7)  gen_kernel(N,8)  \
        gen_kernel(N,9)  gen_kernel(N,10) gen_kernel(N,11) gen_kernel(N,12) gen_kernel(N,13) gen_kernel(N,14) gen_kernel(N,15) gen_kernel(N,16)

    gen_kernels(1) gen_kernels(2)
    gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
    #undef gen_kernels
#endif

#undef gen_kernel
    
}


#endif