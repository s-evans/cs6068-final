#include "blelloch_scan.h"
#include "histogram.h"
#include "limits.h"
#include "radix_sort.h"
#include "relative_offsets.h"
#include "scatter.h"
#include "utils.h"

// define value size and radix sort chunk size
#define RADIX_SORT_TOTAL_BITS ( sizeof( unsigned int ) << 3 )
#define RADIX_SORT_NUM_ITERATIONS ( ( RADIX_SORT_TOTAL_BITS + RADIX_SORT_NUM_BITS - 1 ) / RADIX_SORT_NUM_BITS ) // number of iterations required to perform radix sort
#define BLOCK_SIZE 256

const dim3 block_size( BLOCK_SIZE, 1, 1 );

// generate a mask to keep N lower order bits
const unsigned int mask = ( 1 << RADIX_SORT_NUM_BITS ) - 1;

__global__ void move_values(
        const unsigned int* const d_scatter,
        const unsigned int* const d_relative_offset,
        const unsigned int relative_offset_size,
        const unsigned int* const d_input,
        const unsigned int* const d_input_positions,
        unsigned int* const d_output,
        unsigned int* const d_output_positions,
        const unsigned int input_size,
        const unsigned int bit_offset)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int offset = gridDim.x * blockDim.x;

    while ( idx < input_size ) {
        // shift the integer value into lowest bit order position to prepare for mask
        const unsigned int shifted_value = d_input[idx] >> bit_offset;

        // mask the integer value to get the bin index
        const unsigned int bin = shifted_value & mask;

        // get the row offset for the bin
        const unsigned int row = bin * relative_offset_size;

        // get the output index from adding scatter index and relative offset of input value within its bin
        const unsigned int output_idx = d_scatter[bin] + d_relative_offset[row + idx];

        // move elements from the input arrays into the output arrays
        d_output[output_idx] = d_input[idx];
        d_output_positions[output_idx] = d_input_positions[idx];

        // try next element
        idx += offset;
    }
}

void make_relative_offsets(
        unsigned int* const d_relative_offsets,
        const unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int mask_offset,
        const unsigned int mask,
        cudaStream_t* streams )
{
    const unsigned int relative_offsets_size = blelloch_size( input_size );
    unsigned int row_offset = 0;

    for ( unsigned int value = 0 ; value < RADIX_SORT_NUM_VALS ; ++value ) {

        relative_offsets(
                d_relative_offsets + row_offset,
                d_input,
                input_size,
                value,
                mask_offset,
                mask,
                streams[value] );

        row_offset += relative_offsets_size;
    }
}

#define SWAP(x,y) do { \
    typeof(x) _x = x;  \
    typeof(y) _y = y;  \
    x = _y;            \
    y = _x;            \
} while(0)

void radix_sort(
        unsigned int* const d_input,
        unsigned int* const d_input_positions,
        unsigned int* const d_output,
        unsigned int* const d_output_positions,
        unsigned int const input_size,
        unsigned int* d_histogram,
        unsigned int* d_relative_offset,
        cudaStream_t stream )
{
    cudaStream_t streams[RADIX_SORT_NUM_VALS];
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );
    unsigned int const histogram_size = blelloch_size( RADIX_SORT_NUM_VALS );
    const unsigned int relative_offset_size = blelloch_size( input_size );

    unsigned int* input_vals  = d_input;
    unsigned int* input_pos   = d_input_positions;
    unsigned int* output_vals = d_output;
    unsigned int* output_pos  = d_output_positions;

    for ( unsigned int i = 0 ; i < RADIX_SORT_NUM_VALS ; ++i ) {
        checkCudaErrors( cudaStreamCreate( &streams[i] ) );
    }

    for ( unsigned int mask_offset = 0 ; mask_offset < RADIX_SORT_TOTAL_BITS ; mask_offset += RADIX_SORT_NUM_BITS ) {

        scatter(
                d_histogram,
                histogram_size,
                input_vals,
                input_size,
                mask_offset,
                mask,
                stream );

        make_relative_offsets(
                d_relative_offset,
                input_vals,
                input_size,
                mask_offset,
                mask,
                streams );

        checkCudaErrors( cudaDeviceSynchronize() );

        // TODO: restore

        move_values<<<grid_size, block_size, 0, stream>>>(
                d_histogram,
                d_relative_offset,
                relative_offset_size,
                input_vals,
                input_pos,
                output_vals,
                output_pos,
                input_size,
                mask_offset );

        checkCudaErrors( cudaDeviceSynchronize() );

        SWAP( input_vals, output_vals );
        SWAP( input_pos, output_pos );
    }

    if ( RADIX_SORT_NUM_ITERATIONS % 2 == 0 ) {
        checkCudaErrors( cudaMemcpyAsync(
                    d_output,
                    d_input,
                    sizeof( *d_input ) * input_size,
                    cudaMemcpyDeviceToDevice, 
                    streams[0] ) );

        checkCudaErrors( cudaMemcpyAsync(
                    d_output_positions,
                    d_input_positions,
                    sizeof( *d_output_positions ) * input_size,
                    cudaMemcpyDeviceToDevice, 
                    streams[1] ) );
    }

    for ( unsigned int i = 0 ; i < RADIX_SORT_NUM_VALS ; ++i ) {
        checkCudaErrors( cudaStreamSynchronize( streams[i] ) );
        checkCudaErrors( cudaStreamDestroy( streams[i] ) );
    }
}

