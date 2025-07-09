#pragma once
#include <cstdint>

__host__ __device__ void multiply_rolled(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height);
__host__ __device__ void multiply_rolled_karatsuba(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t height);