#include "vec.hpp"

#ifdef TEST_GROUP_MEMORY_VEC

void group::memory::vec::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/group/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER
    group::memory::vec::global_to_register::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED
    group::memory::vec::global_to_shared::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER
    group::memory::vec::shared_to_register::tests(device, command_queue, results);
#endif

}

#endif

