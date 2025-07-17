#pragma once

#include <cstdint>

#include "../utils/constants.hpp"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_rolled.cuh"
#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK
#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

__global__ void multiply_hybrid_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    multiply_thread(field_element_a, field_element_b, destination, num_bits, threadIdx.x, blockIdx.x);
}

__global__ void multiply_then_add_kernel(const uint32_t* field_element_a, const uint32_t* field_element_b, uint32_t* destination, uint32_t num_bits) {
    __shared__ uint32_t batch_product[BITS_WIDTH];
    multiply_thread(field_element_a, field_element_b, batch_product, num_bits, threadIdx.x, blockIdx.x);
	__syncthreads();
    atomicXor(destination + threadIdx.x, batch_product[threadIdx.x]);
}

__global__ void composition_then_add_kernel(const uint32_t* field_elements, uint32_t* destination, uint32_t num_bits, uint32_t composition_size) {
    uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
    uint32_t num_elements = gridDim.x;

	if(bid == 0) {
		destination[tid] = 0;
	}
	
    __shared__ uint32_t batch_product[BITS_WIDTH];
	batch_product[tid] = 0;
	__syncthreads();
	batch_product[tid] = field_elements[bid * num_bits + tid];
	__syncthreads();

	//printf("%d\n", field_elements[bid * num_bits + tid]);
	if(tid == 0 && bid == 0) {
		printf("multilinear_evaluations[0] = %u\n", field_elements[0]);
	}

    for(int i = 1; i < composition_size; i++) {
		//printf("multiply_thread field_eleemnts + %d, num_bits=%d, tid=%d\n", i*num_elements*num_bits + bid*num_bits, num_bits, tid);
        multiply_thread(batch_product, field_elements + i*num_elements*num_bits + bid*num_bits, batch_product, num_bits, tid, 0);
		__syncthreads();
    }

    atomicXor(destination + tid, batch_product[tid]);
}

__global__ void interpolation_then_composition_then_add(const uint32_t* batches, const uint32_t* coefficient, uint32_t* destination, uint32_t num_bits, uint32_t num_batches, uint32_t composition_size) { // calculate si(xi)
    uint32_t tid = threadIdx.x;
	uint32_t idx = tid + blockIdx.x * blockDim.x;// 127 + 128 * num_batch_rows / 2
    
    __shared__ uint32_t xor_of_halves[BITS_WIDTH];
	__shared__ uint32_t folded_points[BITS_WIDTH];
	__shared__ uint32_t composition[BITS_WIDTH];

	composition[tid] = 0;
	xor_of_halves[tid] = 0;
	folded_points[tid] = 0;

	if(blockIdx.x == 0) {
		destination[tid] = 0;
	}

	__syncthreads();
	//composition[tid] = 0xFFFFFFFF;

	for(int j = 0; j < composition_size; j++) {
		/*if(idx == 0) {
			printf("fine coef[0] = %u\n", coefficient[0]);
		}*/
		const uint32_t* lower_batches = batches + j * num_batches * BITS_WIDTH;
		
		//xor_of_halves[tid] = lower_batches[idx] ^ lower_batches[idx + num_batches * BITS_WIDTH / 2];
		xor_of_halves[tid] = batches[idx + j*num_batches*BITS_WIDTH] ^ batches[idx + j*num_batches*BITS_WIDTH + num_batches*BITS_WIDTH/2];
		folded_points[tid] = 0;

		__syncthreads();
		if(tid * INTERPOLATION_BITS_WIDTH < num_bits) {
			int i = tid * INTERPOLATION_BITS_WIDTH;
			multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(xor_of_halves + i, coefficient, folded_points + i);
		}
		
		__syncthreads();
		folded_points[tid] = folded_points[tid] ^ lower_batches[idx];
		__syncthreads();
		if(j > 0) {
			multiply_thread(composition, folded_points, composition, num_bits, tid, 0);
		} else {
			composition[tid] = folded_points[tid];
		}
		__syncthreads();
	}
	
	atomicXor(destination + tid, composition[tid]);
}

template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
void compute_compositions_fine( // evaluates Si(Xi) at multiple points and gets the claimed sum
	const uint32_t* multilinear_evaluations, // d x 2^n table representing the 3 multiplied hypercubes
	uint32_t* multilinear_products_sums, 
	uint32_t* folded_products_sums,
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],
	const uint32_t num_batch_rows,
	const cudaStream_t streams[INTERPOLATION_POINTS + 1]
) {

	cudaMemset(multilinear_products_sums, 0, BITS_WIDTH * sizeof(uint32_t));
	cudaMemset(folded_products_sums, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

	//printf("launch %d blocks of %d threads", num_batch_rows, BITS_WIDTH);
	composition_then_add_kernel<<<num_batch_rows, BITS_WIDTH>>>(multilinear_evaluations, multilinear_products_sums, BITS_WIDTH, COMPOSITION_SIZE);	

	for(int i = 0; i < INTERPOLATION_POINTS; i++) {
		const uint32_t* coefficient = coefficients + BITS_WIDTH * i;
		uint32_t* destination = folded_products_sums + BITS_WIDTH * i;

		interpolation_then_composition_then_add
			<<<num_batch_rows / 2, BITS_WIDTH>>>(multilinear_evaluations, coefficient, destination, BITS_WIDTH, num_batch_rows, COMPOSITION_SIZE);	
	}

	check(cudaDeviceSynchronize());
}

template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
__global__ void compute_compositions( // evaluates Si(Xi) at multiple points and gets the claimed sum
	// organized by multiple 32x128 batches for P1, followed by many 32x128 for P2, etc
	const uint32_t* multilinear_evaluations, // d x 2^n table representing the 3 multiplied hypercubes

	// 32x128 representing sum across all same index in a batch. 
	// outside of this function, all the elements of this will get summed up in the end to get the claimed value for Si(Xi) 
	uint32_t* multilinear_products_sums, 

	// 32x128xCOMPOSITION_SIZE representing folded sums for different interpolation points 
	// same thing as multilinear_products_sums, this is batched in 32 different values but will get summed
	// up to a single binary tower element later
	uint32_t* folded_products_sums,

	// interpolation points we are plugging in to find Si(Xi)
	// remember that we aren't deriving Si(Xi) ourselves,
	// we are finding the value of it at COMPOSITION_SIZE+1 points for the verifier
	// to Lagrange interpolate
	// the same point is repeated 32 times to allow for batch multiplication
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],

	// number of batches in a single multilinear polynomial table
	const uint32_t num_batch_rows,
	const uint32_t active_threads,
	const uint32_t active_threads_folded
) {
	//printf("here\n");
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	uint32_t folded_products_sums_this_thread[INTERPOLATION_POINTS * BITS_WIDTH];

	uint32_t multilinear_products_sums_this_thread[BITS_WIDTH];

	memset(folded_products_sums_this_thread, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

	memset(multilinear_products_sums_this_thread, 0, BITS_WIDTH * sizeof(uint32_t));

	//if(tid == 0) {
		//printf("regular multilinear_evaluations[%d] = %u\n", 0, multilinear_evaluations[0]);
		//printf("compute_compositions_fine num_batch_rows=%d coef[0] = %u\n", num_batch_rows, coefficients[0]);
	//}

	if(tid == 0) {
		printf("regular multilinear_evaluations[0] = %u\n", multilinear_evaluations[0]);
	}

	for (uint32_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
		uint32_t this_multilinear_product[BITS_WIDTH];

		// finding the claimed sum P(000) + P(001) + P(010) + ... + P(110) + P(111)
		evaluate_composition_on_batch_row( 
			multilinear_evaluations + BITS_WIDTH * row_idx, // the row_idx'th batch 
			this_multilinear_product, // destination for p1p2p3...pd
			COMPOSITION_SIZE, // =d
			EVALS_PER_MULTILINEAR // number of elements in multilinear (2^n) to define the striding of composition
		);

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			multilinear_products_sums_this_thread[i] ^= this_multilinear_product[i]; // add to running batch sums
		}


		// folding to find Si(Xi) = P(Xi, 0, 0) + P(Xi, 0, 1) + P(Xi, 1, 0) + P(Xi, 1, 1) for multiple Xi
		uint32_t num_batch_rows_to_fold = num_batch_rows / 2;

		if (row_idx < num_batch_rows_to_fold) {
			// Fold each batch in the batch row
			// for each interpolation point (INTERPOLATION_POINTS)
			// independent for each polynomial (*COMPOSITION_SIZE)
			// organized by COMPOSITION_SIZE * INTERPOLATION_POINTS * BITS_WIDTH * hypercube idx + INTERPOLATION_POINTS * BITS_WIDTH * interpolation point idx
			uint32_t folded_batch_row[INTERPOLATION_POINTS * COMPOSITION_SIZE * BITS_WIDTH];

			// Fold this batch with the corresponding one
			for (int column_idx = 0; column_idx < COMPOSITION_SIZE; ++column_idx) {
				uint32_t batches_fitting_into_original_column = EVALS_PER_MULTILINEAR / 32;

				// starting index of half starting with 0 ("lower")
				// batched multilinear_evaluations pointer + (ints per multilinear * composition idx) + (batch idx * batch size)
				const uint32_t* lower_batch =
					multilinear_evaluations +
					BITS_WIDTH * (batches_fitting_into_original_column * column_idx + row_idx);
				// upper batch = lower batch + batch size * batches per hypercube / 2
				const uint32_t* upper_batch = lower_batch + BITS_WIDTH * num_batch_rows_to_fold;
				
				// for each interpolation point and multilinear polynomial, fold the upper batch with the lower batch to find Si(Xi) where Xi is the ith interpolation point
				// and save the fold result to folded_batch_row
				for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
					/*if(tid == 0) {
						printf("reg coef[0] = %u\n", *(coefficients + BITS_WIDTH * interpolation_point));
					}*/
					fold_batch( // 3%
						lower_batch,
						upper_batch,
						// folded_batch_row ptr + batch size * hypercube idx * num interpolation points + batch size * interpolation point idx
						folded_batch_row + BITS_WIDTH * (column_idx * INTERPOLATION_POINTS + interpolation_point),
						// ith interpolation point repeated a bunch of times for the entire batch
						// makes bitsliced multiplications easier
						coefficients + BITS_WIDTH * interpolation_point, 
						true
					);
				}
			}

			// Take the folded batches and evaluate the compositions on them
			// find p1p2....pd at each point for each folded polynomial
			uint32_t this_interpolation_point_product_batch[BITS_WIDTH];
			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				// composition batch for this interpolation point.
				
				// find the product of each hypercube batch. (p1p2...pd) 
				evaluate_composition_on_batch_row( // THIS IS THE BIGGEST SLOWDOWN
					// start at the 1st batch in the fold result for that point
					folded_batch_row + BITS_WIDTH * interpolation_point, // starting batch
					this_interpolation_point_product_batch, // destination 
					COMPOSITION_SIZE, // number of batches to multiply (number of multilinear polynomials together)
					INTERPOLATION_POINTS * 32 // stride to let the algorithm determine whcih batches to multiply
				);

				// Add this product to the sum of all products taken by the thread
				// pointer to the sum for this interpolation point
				uint32_t* this_interpolation_point_sum_location =
					folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

				for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
					// add this batch to the sum for this interpolation point
					// atomic not needed; this is still the result just for this thread's batches
					this_interpolation_point_sum_location[i] ^= this_interpolation_point_product_batch[i];
				}
			}
		}
	}

	// accumulate the claimed sum P(000)+P(001)+...P(111)
	if (tid < active_threads) {
		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			// accumulate all thread sums to a single 32x128 element-wise sum
			// we are changing an array shared across all threads so we have to be careful
			// and use atomic operations
			// because of bitslicing it's just a bitwise XOR for each position
			atomicXor(multilinear_products_sums + i, multilinear_products_sums_this_thread[i]); // TODO instead of atomic xor may speedup by a few %
		}
	}

	// accumulate the interpolation sums P(Xi, 0, 0)...P(Xi, 1, 1)
	if (tid < active_threads_folded) {
		// do it for each interpolation point
		for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
			// accumulate thread sums to single 32x128 element-wise sum, basically identical to the last accumulation
			// use atomic operations
			uint32_t* batch_to_copy_to = folded_products_sums + BITS_WIDTH * interpolation_point;
			uint32_t* batch_to_copy_from = folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

			for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
				atomicXor(batch_to_copy_to + i, batch_to_copy_from[i]);
			}
		}
	}
}

__global__ void fold_large_list_halves(
	uint32_t* source,
	uint32_t* destination,
	uint32_t coefficient[BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t src_evals_per_column,
	const uint32_t dst_evals_per_column,
	const uint32_t num_cols
);