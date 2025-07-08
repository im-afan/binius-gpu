#include <cstdint> 
#include <stdio.h>
#include "binary_tower_rolled.cuh"
#include "mul_cache.cuh"

__host__ __device__ void multiply_rolled(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    if(num_bits == 1) {
        destination[0] = field_element_a[0] & field_element_b[0];
    } else if((1 << num_bits) == 16) {
        //printf("here\n");
        for(int i = 0; i < 32; i++) {
            uint32_t a = 0, b = 0, c = 0;
            for(int j = 0; j < num_bits; j++) {
                a ^= (field_element_a[j] & (1 << i)) >> i << j;
                b ^= (field_element_b[j] & (1 << i)) >> i << j;
            }
            //printf("mul_cache[%d][%d]\n", a, b);
            c = mul_cache[a][b];
            for(int j = 0; j < num_bits; j++) {
                destination[j] ^= (c & (1 << j)) >> j << i;
            }
        }
        //printf("here after\n");
    } else {
        uint32_t num_bits_half = num_bits >> 1;
        uint32_t num_bits_quarter = num_bits >> 2;

        uint32_t* z0 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z3 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2_alpha = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* alpha = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* La_Ra = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* Lb_Rb = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        /*uint32_t z3[num_bits_half];
        uint32_t z2[num_bits_half];
        uint32_t z2_alpha[num_bits_half]; // z2 * x_{k-1}
        uint32_t alpha[num_bits_half]; // x_{k-1}
        uint32_t La_Ra[num_bits_half];
        uint32_t Lb_Rb[num_bits_half];*/
        
        multiply_rolled(field_element_a, field_element_b, z0, num_bits_half);
        multiply_rolled(field_element_a + num_bits_half, field_element_b + num_bits_half, z2, num_bits_half);

        for(int i = 0; i < num_bits_half; i++) {
            La_Ra[i] = field_element_a[i] ^ field_element_a[i + num_bits_half]; 
            Lb_Rb[i] = field_element_b[i] ^ field_element_b[i + num_bits_half]; 
            if(i == num_bits_quarter) alpha[i] = 0xFFFFFFFF;
            else alpha[i] = 0;
        } 

        multiply_rolled(La_Ra, Lb_Rb, z3, num_bits_half);
        multiply_rolled(alpha, z2, z2_alpha, num_bits_half);

        // A*B = LaLb + (LaRb + LbRa)xk + RaRb(x_{k-1}x_k + 1)
        // = z0 + z2 + (z1 + z2*x_{k-1}) * x_k

        for(int i = 0; i < num_bits; i++) {
            destination[i] = 0;
            if(i < num_bits_half) {
                destination[i] ^= z0[i] ^ z2[i];
            }
            if(i >= num_bits_half) {
                destination[i] ^= z3[i - num_bits_half] ^ z0[i - num_bits_half] ^ z2[i - num_bits_half] ^ z2_alpha[i - num_bits_half];
            }
        }
    }        
}