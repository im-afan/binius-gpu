#include <cstdint> 
#include <stdio.h>
#include "binary_tower_rolled.cuh"

__host__ __device__ void multiply_alpha(const uint32_t* field_element, uint32_t* destination, uint32_t num_bits) {
    // z2 * x_{k-1} in F_{2^k}
    // let L + R*x_{k-1} = z2
    // (L + R*x_{k-1}) * x_{k-1} = L*x_{k-1} + R*(x_{k-1}*x_{k-2}+1)
    // =  R + x_{k-1}*(L+R*x_{k-2})
    // = R + x_{k-1}*(L+multiply_alpha(r, bits/2))
    //printf("multiply_alpha\n");
    if(num_bits == 1) {
        destination[0] = field_element[0];
    } else{
        uint32_t num_bits_half = num_bits >> 1;
        
        multiply_alpha(field_element + num_bits_half, destination + num_bits_half, num_bits_half); 

        for(int i = 0; i < num_bits; i++) {
            if(i < num_bits_half) {
                destination[i] = field_element[i + num_bits_half];
            } else {
                destination[i] = destination[i] ^ field_element[i - num_bits_half];
            }
        }
    }
}

__host__ __device__ void multiply_rolled(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    if(num_bits == 1) {
        destination[0] = field_element_a[0] & field_element_b[0];
    } else {
        uint32_t num_bits_half = num_bits >> 1;

        uint32_t* z0 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z1 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2_alpha = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        
        multiply_rolled(field_element_a, field_element_b, z0, num_bits_half);
        multiply_rolled(field_element_a + num_bits_half, field_element_b + num_bits_half, z2, num_bits_half);
        multiply_rolled(field_element_a, field_element_b + num_bits_half, z1, num_bits_half);
        multiply_rolled(field_element_a + num_bits_half, field_element_b, z1, num_bits_half);
        multiply_alpha(z2, z2_alpha, num_bits_half);

        // A*B = LaLb + (LaRb + LbRa)xk + RaRb(x_{k-1}x_k + 1)
        // = z0 + z2 + (z1 + z2*x_{k-1}) * x_k

        for(int i = 0; i < num_bits; i++) {
            if(i < num_bits_half) {
                destination[i] ^= z0[i] ^ z2[i];
            } else {
                destination[i] ^= z1[i - num_bits_half] ^ z2_alpha[i - num_bits_half];
            }
        }
    }        
}

__host__ __device__ void multiply_rolled_karatsuba(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    if(num_bits == 1) {
        destination[0] = field_element_a[0] & field_element_b[0];
    } else {
        uint32_t num_bits_half = num_bits >> 1;

        uint32_t* z0 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z3 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2_alpha = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        
        multiply_rolled(field_element_a, field_element_b, z0, num_bits_half);
        multiply_rolled(field_element_a + num_bits_half, field_element_b + num_bits_half, z2, num_bits_half);

        for(int i = 0; i < num_bits_half; i++) {
            destination[i] = field_element_a[i] ^ field_element_a[i + num_bits_half]; 
            destination[i + num_bits_half] = field_element_b[i] ^ field_element_b[i + num_bits_half]; 
        } 

        multiply_rolled(destination, destination + num_bits_half, z3, num_bits_half);
        multiply_alpha(z2, z2_alpha, num_bits_half);

        // A*B = LaLb + (LaRb + LbRa)xk + RaRb(x_{k-1}x_k + 1)
        // = z0 + z2 + (z1 + z2*x_{k-1}) * x_k

        for(int i = 0; i < num_bits; i++) {
            if(i < num_bits_half) {
                destination[i] = z0[i] ^ z2[i];
            } else {
                destination[i] = z3[i - num_bits_half] ^ z0[i - num_bits_half] ^ z2[i - num_bits_half] ^ z2_alpha[i - num_bits_half];
            }
        }
    }        
}

__host__ __device__ void multiply_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t tower_height, uint32_t num_bits) {
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int start_idx = threadIdx.x + blockDim.x * gridDim.x;

    for(int i = tid; i < num_bits; i++) {

    }
}