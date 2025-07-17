#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include "../core/kernels.cuh"
#include "../core/core.cuh"

void test_composition_then_add_kernel() {
    uint32_t x[128*4];// = malloc(128 * 4 * sizeof(uint32_t));
    uint32_t solution[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t output[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t composition[128*2];// = malloc(128 * 2 * sizeof(uint32_t));

    uint32_t* x_d;
    uint32_t* output_d;

    cudaMalloc((void**) &x_d, 128 * 4 * sizeof(uint32_t));
    cudaMalloc((void**) &output_d, 128*sizeof(uint32_t));

    for(int i = 0; i < 128 * 4; i++) {
        if(i < 128) {
            solution[i] = 0;
            output[i] = 0;
        }
        if(i < 2*128) {
            composition[i] = 0xFFFFFFFF;
        }
        x[i] = std::rand();
    }

    cudaMemcpy(x_d, x, 128*4*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output, 128*sizeof(uint32_t), cudaMemcpyHostToDevice);

    for(int j = 0; j < 2; j++) {
        //printf("%d %d\n", x[j*128], x[j*128+256]);
        evaluate_composition_on_batch_row(x + j * 128, composition + j * 128, 2, 64);
    }

    for(int j = 0; j < 2; j++) {
        for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
            solution[i] ^= composition[i + j*128]; // add to running batch sums
        }
    }

    composition_then_add_kernel<<<2, 128>>>(x_d, output_d, 128, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, 128*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 128; i++) {
        REQUIRE(output[i] == solution[i]);
    }
}

TEST_CASE("composition_then_add_kernel") {
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
    test_composition_then_add_kernel();
}

void test_interpolation_then_composition_then_add() {
    uint32_t x[128*4];// = malloc(128 * 4 * sizeof(uint32_t));
    uint32_t coefficient[128];
    uint32_t solution[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t output[128];// = malloc(128 * sizeof(uint32_t)); 
    uint32_t composition[128];// = malloc(128 * 2 * sizeof(uint32_t));
    uint32_t folded[128*2];// = malloc(128 * 2 * sizeof(uint32_t));

    uint32_t* x_d;
    uint32_t* coefficient_d;
    uint32_t* output_d;

    check(cudaMalloc((void**) &x_d, 128 * 4 * sizeof(uint32_t)));
    check(cudaMalloc((void**) &output_d, 128*sizeof(uint32_t)));
    check(cudaMalloc((void**) &coefficient_d, 128*sizeof(uint32_t)));

    for(int i = 0; i < 128 * 4; i++) {
        if(i < 128) {
            solution[i] = 0;
            output[i] = 0;
            coefficient[i] = 0;
            composition[i] = 0xFFFFFFFF;
        }
        x[i] = std::rand();
    }

    check(cudaMemcpy(x_d, x, 128*4*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output_d, output, 128*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(coefficient_d, coefficient, 128*sizeof(uint32_t), cudaMemcpyHostToDevice));

    for(int j = 0; j < 2; j++) {
        fold_batch(x + j*256, x + 128 + j*256, folded + j*128, coefficient, true); 
    }
    evaluate_composition_on_batch_row(folded, composition, 2, 32);
    for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
        solution[i] ^= composition[i]; // add to running batch sums
    }

    interpolation_then_composition_then_add<<<1, 128>>>(x_d, coefficient_d, output_d, 128, 2, 2);
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(output, output_d, 128*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for(int i = 0; i < 128; i++) {
        REQUIRE(output[i] == solution[i]);
    }

}

TEST_CASE("interpolation_then_composition_then_add_kernel") {
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
    test_interpolation_then_composition_then_add(); 
}