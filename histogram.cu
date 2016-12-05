#include "utils.h"
#include "histogram.h"

__global__
void histogram_kernel(
        unsigned int* const histogram,
        const unsigned char* const input,
        const unsigned int input_size )
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
        const unsigned char val = input[idx];

        // do atomic increment of the bin in shared memory
        atomicAdd( &shared_histogram[val], 1 );
    }

    __syncthreads();

    // increment global memory with count in shared memory for each bin
    atomicAdd( &histogram[tid], shared_histogram[tid] );
}

void histogram(
        unsigned int* d_histogram,
        unsigned int histogram_size,
        unsigned char* d_input,
        unsigned int input_size)
{
    const dim3 block_size( histogram_size, 1, 1 );
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );
    const unsigned int shared_size = sizeof( *d_histogram ) * histogram_size;

    histogram_kernel<<<grid_size, block_size, shared_size, 0>>>( d_histogram, d_input, input_size );
}
