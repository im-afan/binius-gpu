#pragma once
#include <cstdint>

__host__ __device__ void multiply_rolled(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height);
__host__ __device__ void multiply_rolled_karatsuba(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height);
__global__ void multiply_kernel_decompose(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits);
__global__ void multiply_kernel_compose(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height, uint32_t num_bits);
__global__ void multiply_karatsuba_decompose_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height, uint32_t num_bits);
__global__ void multiply_karatsuba_compose_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height, uint32_t num_bits);
void multiply_parallel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits);

void multiply_hybrid(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination);

//template <uint32_t HEIGHT>

//__global__ void multiply_hybrid_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits);
//__global__ void multiply_then_add_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits);
void multiply_unrolled_on_device(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination);
void multiply_hybrid_inplace(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination);

__global__ void multiply_unrolled_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination);

//__global__ void composition_then_add_kernel(const uint32_t* field_elements, uint32_t* destination, uint32_t num_bits, uint32_t composition_size);

__device__ void multiply_thread(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits, int tid, int bid);