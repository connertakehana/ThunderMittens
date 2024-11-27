#include "global_to_shared.hpp"
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED
#include "MetalSingle.hpp"



template<typename T>
struct group_vec_gts_load_store {
    using dtype = T;
    template<int L, int NW, typename... args> using valid = std::bool_constant<NW <= 4 && L <= 64>; // this is warp-level
    
    
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_vec_gts_loadstore=bf16" :
                                                      std::is_same_v<T, half> ? "group_vec_gts_loadstore=half" :
                                                                                "group_vec_gts_loadstore=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_vec_gts_loadstore_bf16" :
                                                        std::is_same_v<T, half> ? "group_vec_gts_loadstore_half" :
                                                                                  "group_vec_gts_loadstore_float";
    template<int H, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
};

void group::memory::vec::global_to_shared::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    general_sweep_size_1d<group_vec_gts_load_store<float>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<group_vec_gts_load_store<float>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<group_vec_gts_load_store<float>, SIZE, 4>::run(device, command_queue, results);
    
    general_sweep_size_1d<group_vec_gts_load_store<half>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<group_vec_gts_load_store<half>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<group_vec_gts_load_store<half>, SIZE, 4>::run(device, command_queue, results);
    
    general_sweep_size_1d<group_vec_gts_load_store<bf16>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<group_vec_gts_load_store<bf16>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<group_vec_gts_load_store<bf16>, SIZE, 4>::run(device, command_queue, results);
}

#endif


////
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::memory::vec::global_to_shared::tests(device, command_queue, results);
//}
