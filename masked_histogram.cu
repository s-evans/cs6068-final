#include "masked_histogram.h"

__global__
void masked_histogram_kernel(
        unsigned int* const histogram,
        const unsigned int* const input,
        const unsigned int input_size,
        const unsigned int mask_offset,
        const unsigned int mask )
{
    extern __shared__ unsigned int shared_histogram[];
    const unsigned int tid = threadIdx.x;

    // initialize counts to zero
    shared_histogram[tid] = 0;

    // wait until the entire shared memory buffer is initialized
    __syncthreads();

    // get the index this thread is responsible for
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // validate against input size
    if ( idx < input_size ) {

        // get the value / bin
        const unsigned int val = ( input[idx] >> mask_offset ) & mask;

        // do atomic increment of the bin in shared memory
        atomicAdd( &shared_histogram[val], 1 );

    }

    // wait until shared memory is finished being used by all threads in the block
    __syncthreads();

    // increment global memory with count in shared memory for each bin
    atomicAdd( &histogram[tid], shared_histogram[tid] );
}

void masked_histogram(
        unsigned int* const d_histogram,
        const unsigned int histogram_size,
        const unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int mask_offset,
        const unsigned int mask,
        cudaStream_t stream )
{
    const dim3 block_size( histogram_size, 1, 1 );
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );
    const unsigned int shared_size = sizeof( *d_histogram ) * histogram_size;

    masked_histogram_kernel<<<grid_size, block_size, shared_size, stream>>>( d_histogram, d_input, input_size, mask_offset, mask );
}

