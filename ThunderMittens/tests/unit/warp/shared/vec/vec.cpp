#include "vec.hpp"

#ifdef TEST_WARP_SHARED_VEC

void warp::shared::vec::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_VEC_CONVERSIONS
    warp::shared::vec::conversions::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_SHARED_VEC_MAPS
    warp::shared::vec::maps::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_SHARED_VEC_REDUCTIONS
    warp::shared::vec::reductions::tests(device, command_queue, results);
#endif

}

#endif
