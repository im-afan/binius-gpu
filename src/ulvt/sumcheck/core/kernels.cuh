#include <cstdint>

#include "../utils/constants.hpp"

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
			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				// composition batch for this interpolation point.
				uint32_t this_interpolation_point_product_batch[BITS_WIDTH];
				
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