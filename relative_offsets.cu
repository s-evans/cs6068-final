#include "relative_offsets.h"
#include "blelloch_scan.h"

#define BLOCK_SIZE 512

const dim3 block_size( BLOCK_SIZE, 1, 1 );

__global__ void predicate_map(
        unsigned int* const d_output,
        const unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int predicate_value,
        const unsigned int mask_offset,
        const unsigned int mask )
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( idx < input_size ) {
        const unsigned int shifted_value = d_input[idx] >> mask_offset;
        const unsigned int masked_value = shifted_value & mask;
        d_output[idx] = ( masked_value == predicate_value ? 1 : 0 );
        return;
    } 

    d_output[idx] = 0;
}

void relative_offsets(
        unsigned int* const d_relative_offsets,
        const unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int predicate_value,
        const unsigned int mask_offset,
        const unsigned int mask,
        cudaStream_t stream )
{
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );

    predicate_map<<<grid_size, block_size, 0, stream>>>(
            d_relative_offsets,
            d_input,
            input_size,
            predicate_value,
            mask_offset,
            mask );

    blelloch_scan( d_relative_offsets, blelloch_size( input_size ), stream );
}
