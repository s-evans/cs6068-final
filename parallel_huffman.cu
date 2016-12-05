#include "blelloch_scan.h"
#include "debug_print.h"
#include "histogram.h"
#include "huffman_tree.h"
#include "limits.h"
#include "parallel_huffman.h"
#include "radix_sort.h"
#include "utils.h"

#define STREAM_COUNT 5

__global__ void init_positions(
        unsigned int* const d_positions )
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    d_positions[idx] = idx;
}

void parallel_huffman_encode(
        const unsigned char* const h_input_buffer,
        unsigned int const input_size,
        unsigned char* const h_output_buffer,
        unsigned int& output_size )
{
    cudaStream_t streams[STREAM_COUNT];

    for ( int i = 0 ; i < STREAM_COUNT ; ++i ) {
        checkCudaErrors( cudaStreamCreate( &streams[i] ) );
    }

    unsigned char* d_input;
    checkCudaErrors( cudaMalloc( &d_input, sizeof( *d_input ) * input_size ) );
    checkCudaErrors( cudaHostRegister( static_cast<void*>( const_cast<unsigned char*>( h_input_buffer ) ), input_size, cudaHostRegisterMapped ) );
    checkCudaErrors( cudaMemcpyAsync( d_input, h_input_buffer, input_size, cudaMemcpyHostToDevice, streams[0] ) );

    unsigned char* d_output;
    checkCudaErrors( cudaMalloc( &d_output, sizeof( *d_output ) * input_size ) );
    checkCudaErrors( cudaMemsetAsync( d_output, 0, input_size, streams[1] ) );

    unsigned int* d_histogram;
    const unsigned int histogram_count = 1 << ( sizeof( *d_input ) << 3 );
    const unsigned int histogram_size = sizeof( *d_histogram ) << ( sizeof( *d_input ) << 3 );
    checkCudaErrors( cudaMalloc( &d_histogram, histogram_size ) );
    checkCudaErrors( cudaMemsetAsync( d_histogram, 0, histogram_size, streams[2] ) );

    unsigned int* d_input_positions;
    checkCudaErrors( cudaMalloc( &d_input_positions, histogram_size ) );
    init_positions<<<1, histogram_count, 0, streams[3]>>>( d_input_positions );

    unsigned int* d_sorted_histogram;
    checkCudaErrors( cudaMalloc( &d_sorted_histogram, histogram_size ) );

    unsigned int* d_relative_offsets;
    const unsigned int radix_relative_offsets_count = blelloch_size( histogram_count ) * RADIX_SORT_NUM_VALS;
    checkCudaErrors( cudaMalloc( &d_relative_offsets, radix_relative_offsets_count * sizeof( *d_relative_offsets ) ) );

    unsigned int* d_output_positions;
    checkCudaErrors( cudaMalloc( &d_output_positions, histogram_size ) );

    unsigned int* d_radix_histogram;
    const unsigned int radix_blelloch_number = blelloch_size( RADIX_SORT_NUM_VALS );
    checkCudaErrors( cudaMalloc( &d_radix_histogram, radix_blelloch_number * sizeof( *d_radix_histogram ) ) );

    checkCudaErrors( cudaStreamSynchronize( streams[0] ) );

    histogram( d_histogram, histogram_count, d_input, input_size, streams[2] );

    checkCudaErrors( cudaDeviceSynchronize() );

    /* std::cerr << "unsorted histogram" << std::endl; */
    /* debug_print( d_histogram, histogram_count ); */

    /* std::cerr << "unsorted symbols" << std::endl; */
    /* debug_print( d_input_positions, histogram_count ); */

    radix_sort(
            d_histogram,
            d_input_positions,
            d_sorted_histogram,
            d_output_positions,
            histogram_count,
            d_radix_histogram,
            d_relative_offsets,
            streams[0] );

    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaFree( d_histogram ) );
    checkCudaErrors( cudaFree( d_input_positions ) );
    checkCudaErrors( cudaFree( d_radix_histogram ) );
    checkCudaErrors( cudaFree( d_relative_offsets ) );

    /* std::cerr << "sorted histogram" << std::endl; */
    /* debug_print( d_sorted_histogram, histogram_count ); */

    /* std::cerr << "sorted symbols" << std::endl; */
    /* debug_print( d_output_positions, histogram_count ); */

    huffman_encode(
            d_output,
            &output_size,
            d_input,
            input_size,
            d_sorted_histogram,
            d_output_positions );

    // TODO: output the huffman table along with output data

    checkCudaErrors( cudaHostUnregister( static_cast<void*>( const_cast<unsigned char*>( h_input_buffer ) ) ) );
    checkCudaErrors( cudaHostRegister( static_cast<void*>( h_output_buffer ), output_size, cudaHostRegisterMapped ) );

    checkCudaErrors( cudaMemcpyAsync( h_output_buffer, d_output, output_size, cudaMemcpyDeviceToHost, streams[0] ) );

    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaHostUnregister( static_cast<void*>( h_output_buffer ) ) );

    checkCudaErrors( cudaFree( d_input ) );
    checkCudaErrors( cudaFree( d_output ) );
    checkCudaErrors( cudaFree( d_sorted_histogram ) );
    checkCudaErrors( cudaFree( d_output_positions ) );
}
