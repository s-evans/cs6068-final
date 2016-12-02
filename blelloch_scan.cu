#include "blelloch_scan.h"
#include "utils.h"

#define BLOCK_SIZE 512

const dim3 block_size( BLOCK_SIZE, 1, 1 );

unsigned int blelloch_size ( unsigned int n )
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return ++n;
}

__global__ void do_exclusive_sum_scan_reduce(
        unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int step_size )
{
    // Get indices
    const int right_idx = input_size - 1 - ( ( threadIdx.x + blockDim.x * blockIdx.x ) * ( step_size << 1 ) );
    const int left_idx = right_idx - step_size;

    // Validate indices
    if ( left_idx < 0 || right_idx < 0 ) {
        return;
    }

    // Reduce
    d_input[right_idx] += d_input[left_idx];
}

__global__ void do_exclusive_sum_scan_downsweep(
        unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int step_size )
{
    // Get indices
    const int right_idx = input_size - 1 - ( ( threadIdx.x + blockDim.x * blockIdx.x ) * ( step_size << 1 ) );
    const int left_idx = right_idx - step_size;

    // Validate indices
    if ( left_idx < 0 || right_idx < 0 ) {
        return;
    }

    // Downsweep
    const unsigned int temp = d_input[right_idx];
    d_input[right_idx] += d_input[left_idx];
    d_input[left_idx] = temp;
}

void blelloch_scan(
        unsigned int* const d_input,
        const unsigned int input_size,
        cudaStream_t stream )
{
    unsigned int iteration = 1;
    unsigned int step_size = 1;

    while ( step_size < input_size ) {

        // Reduce grid size each iteration
        const unsigned int thread_count = input_size >> iteration;
        const dim3 grid_size( ( thread_count + block_size.x - 1 ) / block_size.x, 1, 1 );

        // Do sum reduce
        do_exclusive_sum_scan_reduce<<<grid_size, block_size, 0, stream>>>(
                d_input,
                input_size,
                step_size );

        // Increment
        step_size <<= 1;
        ++iteration;
    }

    // Set rightmost value to zero
    checkCudaErrors( cudaMemsetAsync( &d_input[input_size-1], 0, sizeof( *d_input ), stream ) );

    while ( step_size > 0 ) {
        // Increase grid size each iteration
        const unsigned int thread_count = input_size >> iteration;
        const dim3 grid_size( ( thread_count + block_size.x - 1 ) / block_size.x, 1, 1 );

        // Do downsweep
        do_exclusive_sum_scan_downsweep<<<grid_size, block_size, 0, stream>>>(
                d_input,
                input_size,
                step_size );

        // Decrement
        step_size >>= 1;
        --iteration;
    }
}
