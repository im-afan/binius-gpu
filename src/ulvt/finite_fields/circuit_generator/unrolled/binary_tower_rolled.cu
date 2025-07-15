#include <cstdint> 
#include <stdio.h>
#include "binary_tower_rolled.cuh"
#include "../constants.hpp"
#include "binary_tower_unrolled.cuh"
//#include "../../../sumcheck/core/kernels.cuh"

#define HEIGHT 7

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__host__ __device__ void add(const uint32_t* a, const uint32_t* b, uint32_t* destination, uint32_t num_bits) {
    for(int i = 0; i < num_bits; i++) 
        destination[i] = a[i] ^ b[i];
}

__host__ __device__ void add(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* destination, uint32_t num_bits) {
    for(int i = 0; i < num_bits; i++) 
        destination[i] = a[i] ^ b[i] ^ c[i];
}

__host__ __device__ void multiply_alpha(const uint32_t* field_element, uint32_t* destination, uint32_t num_bits) { // todo unrolled
    // z2 * x_{k-1} in F_{2^k}
    // let L + R*x_{k-1} = z2
    // (L + R*x_{k-1}) * x_{k-1} = L*x_{k-1} + R*(x_{k-1}*x_{k-2}+1)
    // =  R + x_{k-1}*(L+R*x_{k-2})
    // = R + x_{k-1}*(L+multiply_alpha(r, bits/2))
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

__host__ __device__ void multiply_alpha_bit(const uint32_t* field_element, uint32_t* destination, uint32_t num_bits, int idx) {
    uint32_t half_num_bits = num_bits >> 1;
    destination[idx] ^= field_element[(idx + half_num_bits) % num_bits];
    if(idx >= half_num_bits && idx != 0)
        multiply_alpha_bit(field_element + half_num_bits, destination + half_num_bits, half_num_bits, idx - half_num_bits);
}

__host__ __device__ void compose_partials(const uint32_t* z0, const uint32_t* z2, const uint32_t* z3, const uint32_t* z2_alpha, uint32_t* destination, uint32_t idx, uint32_t num_bits) {
    int r_idx = idx + (num_bits>>1);
    destination[idx] = z0[idx] ^ z2[idx];
    destination[r_idx] = z3[idx] ^ z2[idx] ^ z0[idx] ^ z2_alpha[idx];
}

__host__ __device__ void multiply_rolled(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) { // TODO unrolled naive multiplication
    if(num_bits == 1) {
        destination[0] = field_element_a[0] & field_element_b[0];
    } else {
        uint32_t num_bits_half = num_bits >> 1;

        uint32_t* z0 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z1a = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z1b = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2 = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        uint32_t* z2_alpha = (uint32_t*)malloc(num_bits_half * sizeof(uint32_t));
        
        multiply_rolled(field_element_a, field_element_b, z0, num_bits_half);
        multiply_rolled(field_element_a + num_bits_half, field_element_b + num_bits_half, z2, num_bits_half);
        multiply_rolled(field_element_a, field_element_b + num_bits_half, z1a, num_bits_half);
        multiply_rolled(field_element_a + num_bits_half, field_element_b, z1b, num_bits_half);
        multiply_alpha(z2, z2_alpha, num_bits_half);

        // A*B = LaLb + (LaRb + LbRa)xk + RaRb(x_{k-1}x_k + 1)
        // = z0 + z2 + (z1 + z2*x_{k-1}) * x_k

        for(int i = 0; i < num_bits; i++) {
            if(i < num_bits_half) {
                destination[i] = z0[i] ^ z2[i];
            } else {
                destination[i] = z1a[i - num_bits_half] ^ z1b[i - num_bits_half] ^ z2_alpha[i - num_bits_half];
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
        
        multiply_rolled_karatsuba(field_element_a, field_element_b, z0, num_bits_half);
        multiply_rolled_karatsuba(field_element_a + num_bits_half, field_element_b + num_bits_half, z2, num_bits_half);

        for(int i = 0; i < num_bits_half; i++) {
            destination[i] = field_element_a[i] ^ field_element_a[i + num_bits_half]; 
            destination[i + num_bits_half] = field_element_b[i] ^ field_element_b[i + num_bits_half]; 
        } 

        multiply_rolled_karatsuba(destination, destination + num_bits_half, z3, num_bits_half);
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

__global__ void multiply_kernel_decompose(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    __shared__ uint32_t a_s[16], b_s[16];
    __shared__ uint32_t ab_s[16*16];

    uint32_t tid = threadIdx.x;
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for(int i = idx; i < num_bits * num_bits; i += stride) {
        uint32_t a_idx = 0, b_idx = 0;

        int tmp_i = i;
        int odd = 0;
        int cnt = 0;
        while(tmp_i > 0) {
            if(odd == 0) {
                a_idx |= (tmp_i & 1) << cnt;
            } else {
                b_idx |= (tmp_i & 1) << cnt;
                cnt++;
            }
            tmp_i >>= 1;
            odd = 1 - odd;
        }
        
        if(a_idx % 16 == 0) {
            b_s[b_idx % 16] = field_element_b[b_idx];
        }
        if(b_idx % 16 == 0) {
            a_s[a_idx % 16] = field_element_a[a_idx];
        }

        __syncthreads();
        
        ab_s[tid] = a_s[a_idx % 16] & b_s[b_idx % 16];

        __syncthreads();
        
        // A*B = LaLb + (LaRb + LbRa)xk + RaRb(x_{k-1}x_k + 1)
        // = z0 + z2 + (z1 + z2*x_{k-1}) * x_k

        if(i % 4 == 0) {
            destination[i / 2] = ab_s[tid] ^ ab_s[tid + 3];
            destination[i / 2 + 1] = ab_s[tid + 1] ^ ab_s[tid + 2] ^ ab_s[tid + 3];
        }

        __syncthreads();
    }
    //}
}

__global__ void multiply_kernel_compose(const uint32_t* decomposed, uint32_t* destination, uint32_t height, uint32_t num_elements) {
    //uint32_t 
    int num_bits = 1 << height;

    uint32_t tid = threadIdx.x;
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;
    uint32_t z2_alpha[128];

    for(int i = idx; i < num_elements / 4; i += stride) {
        const uint32_t* start = decomposed + i*4*num_bits;
        uint32_t* destination_start = destination + i*2*num_bits;

        multiply_alpha(start + 3*num_bits, z2_alpha, num_bits);

        add(start, start + 3*num_bits, destination_start, num_bits);
        add(start + 1*num_bits, start + 2*num_bits, z2_alpha, destination_start + num_bits, num_bits);
    }
}

void multiply_parallel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) { // TODO streaming
    uint32_t* a_d;
    uint32_t* b_d;
    uint32_t* decomposition;
    check(cudaMalloc((void**) &decomposition, num_bits * num_bits * sizeof(uint32_t)));
    check(cudaMalloc((void**) &a_d, num_bits * sizeof(uint32_t)));
    check(cudaMalloc((void**) &b_d, num_bits * sizeof(uint32_t)));
    check(cudaMemcpy(a_d, field_element_a, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b_d, field_element_b, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));

    multiply_kernel_decompose<<<1024, 256>>>(a_d, b_d, decomposition, num_bits); 
    check(cudaDeviceSynchronize());

    int idx = 0;
    int increment = num_bits * num_bits / 2;
    int num_elements = num_bits * num_bits / 4;
    int height = 1;
    while(num_elements > 1) {
        //printf("idx = %d, idx+increment=%d, numel=%d, num_bits=%d\n", idx, idx+increment, num_elements, 1<<height);
        multiply_kernel_compose<<<1024, 128>>>(decomposition + idx, decomposition + idx + increment, height, num_elements);  
        check(cudaDeviceSynchronize());
        idx += increment;
        increment = increment / 2;
        num_elements = num_elements / 4;
        height++;
    }
    //printf("final idx %d out of %d, there are %d bits.\n", idx, num_bits*num_bits, num_bits);
    check(cudaMemcpy(destination, decomposition + idx, num_bits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

//template<int HEIGHT>
__global__ void multiply_unrolled_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination) {
    //uint32_t HEIGHT = 7;
    multiply_unrolled<HEIGHT>(field_element_a, field_element_b, destination);
}

__device__ void multiply_thread(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits, int tid, int bid) {
    int half_num_bits = num_bits >> 1;
    int quarter_num_bits = num_bits >> 2;

    __shared__ uint32_t a_s6[3*64]; // L, R, F (2^6 bits each)
    __shared__ uint32_t b_s6[3*64];
    __shared__ uint32_t a_s5[9*32]; // LL, LR, RL, RR, FL, FR, LF, RF, FF (2^5 bits each)
    __shared__ uint32_t b_s5[9*32];
    __shared__ uint32_t partials5[16 * 32]; // partials at lowest unrolled multiplication level (5 tower height)
    __shared__ uint32_t partials6[4 * 64]; // next level (6 tower height) (composition of partials5)
    // 1728 ints of shared memory
    
    for(int i = 1; i <= 4; i++) {
        int idx = tid + i*blockDim.x;
        if(idx < 16*32) partials5[idx] = 0;
        if(idx < 4*64) partials6[idx] = 0;
    }

    a_s6[tid + bid * 128] = field_element_a[tid + bid * 128];
    b_s6[tid + bid * 128] = field_element_b[tid + bid * 128];
    
    if(tid < 64) {
        a_s6[tid + 128] = field_element_a[tid + bid * 128] ^ field_element_a[tid + 64 + bid * 128];
        b_s6[tid + 128] = field_element_b[tid + bid * 128] ^ field_element_b[tid + 64 + bid * 128];
    }

    __syncthreads();

    a_s5[tid] = a_s6[tid]; // LL, LR, RL, RR
    b_s5[tid] = b_s6[tid];

    if(tid < 64) {
        a_s5[tid + 128] = a_s6[tid + 128]; // FL, FR (fold then left)
        b_s5[tid + 128] = b_s6[tid + 128];
    }

    if(tid < 32) {
        a_s5[tid + 192] = a_s6[tid] ^ a_s6[tid + 32]; // LF
        b_s5[tid + 192] = b_s6[tid] ^ b_s6[tid + 32]; 
    }
    if(tid >= 32 && tid < 64) {
        int tmp_tid = tid - 32;
        a_s5[tmp_tid + 192 + 32] = a_s6[tmp_tid + 64] ^ a_s6[tmp_tid + 64 + 32]; // RF
        b_s5[tmp_tid + 192 + 32] = b_s6[tmp_tid + 64] ^ b_s6[tmp_tid + 64 + 32]; 
    }
    if(tid >= 64 && tid < 96) {
        int tmp_tid = tid - 64;
        a_s5[tmp_tid + 192 + 64] = a_s6[tmp_tid + 128] ^ a_s6[tmp_tid + 128 + 32]; // FF
        b_s5[tmp_tid + 192 + 64] = b_s6[tmp_tid + 128] ^ b_s6[tmp_tid + 128 + 32];
    }

    __syncthreads();

    if(tid < 9) {
        multiply_unrolled<5>(a_s5 + tid*quarter_num_bits, b_s5 + tid*quarter_num_bits, partials5 + tid*quarter_num_bits);
    }

    __syncthreads();

    if(tid < 32) {
        multiply_alpha_bit(partials5 + 1*quarter_num_bits, partials5 + 9*quarter_num_bits, quarter_num_bits, tid);
    }
    if(tid >= 32 && tid < 64) {
        multiply_alpha_bit(partials5 + 3*quarter_num_bits, partials5 + 10*quarter_num_bits, quarter_num_bits, tid-32);
    }
    if(tid >= 64 && tid < 96) {
        multiply_alpha_bit(partials5 + 5*quarter_num_bits, partials5 + 11*quarter_num_bits, quarter_num_bits, tid-64);
    }

    __syncthreads();

    if(tid < 32) {
        compose_partials(partials5, partials5 + quarter_num_bits, partials5 + 6*quarter_num_bits, partials5 + 9*quarter_num_bits, partials6, tid, half_num_bits);
    }
    if(tid >= 32 && tid < 64) {
        int tmp_tid = tid - 32;
        compose_partials(partials5 + 2*quarter_num_bits, partials5 + 3*quarter_num_bits, partials5 + 7*quarter_num_bits, partials5 + 10*quarter_num_bits, partials6+half_num_bits, tmp_tid, half_num_bits);
    }
    if(tid >= 64 && tid < 96) {
        int tmp_tid = tid - 64;
        compose_partials(partials5 + 4*quarter_num_bits, partials5 + 5*quarter_num_bits, partials5 + 8*quarter_num_bits, partials5 + 11*quarter_num_bits, partials6+2*half_num_bits, tmp_tid, half_num_bits);
    }

    __syncthreads();

    if(tid < 64) {
        multiply_alpha_bit(partials6 + half_num_bits, partials6 + 3*half_num_bits, half_num_bits, tid);
    }

    __syncthreads();

    if(tid < 64) {
        compose_partials(partials6, partials6 + half_num_bits, partials6 + 2*half_num_bits, partials6 + 3*half_num_bits, destination + bid*128, tid, num_bits);
    }
}


void multiply_hybrid(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination) {
    uint32_t num_bits = 1 << HEIGHT;
    uint32_t* a_d;
    uint32_t* b_d;
    uint32_t* destination_d;
    check(cudaMalloc(&a_d, num_bits * sizeof(uint32_t)));
    check(cudaMalloc(&b_d, num_bits * sizeof(uint32_t)));
    check(cudaMalloc(&destination_d, num_bits * sizeof(uint32_t)));
    check(cudaMemcpy(a_d, field_element_a, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b_d, field_element_b, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));

    //multiply_hybrid_kernel<<<1, 128>>>(a_d, b_d, destination_d, num_bits);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(destination, destination_d, num_bits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void multiply_hybrid_inplace(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination) {
    uint32_t num_bits = 1 << HEIGHT;
    uint32_t* a_d;
    uint32_t* b_d;
    check(cudaMalloc(&a_d, num_bits * sizeof(uint32_t)));
    check(cudaMalloc(&b_d, num_bits * sizeof(uint32_t)));
    check(cudaMemcpy(a_d, field_element_a, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b_d, field_element_b, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));

    //multiply_hybrid_kernel<<<1, 128>>>(a_d, b_d, a_d, num_bits);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(destination, a_d, num_bits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

//template<int HEIGHT>
void multiply_unrolled_on_device(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination) {
    //uint32_t HEIGHT = 7;
    uint32_t num_bits = 1 << HEIGHT;
    uint32_t* a_d;
    uint32_t* b_d;
    uint32_t* destination_d;
    check(cudaMalloc(&a_d, num_bits * sizeof(uint32_t)));
    check(cudaMalloc(&b_d, num_bits * sizeof(uint32_t)));
    check(cudaMalloc(&destination_d, num_bits * sizeof(uint32_t)));
    check(cudaMemcpy(a_d, field_element_a, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(b_d, field_element_b, num_bits * sizeof(uint32_t), cudaMemcpyHostToDevice));

    multiply_unrolled_kernel<<<1, 1>>>(a_d, b_d, destination_d);

    check(cudaDeviceSynchronize());
    check(cudaMemcpy(destination, destination_d, num_bits * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}