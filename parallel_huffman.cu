#include "parallel_huffman.h"
#include "blelloch_scan.h"
#include "radix_sort.h"
#include "histogram.h"
#include "utils.h"
#include "limits.h"
#include "debug_print.h"

#define STREAM_COUNT 5
#define BLOCK_SIZE 256 // number of elements per block

const dim3 block_size( BLOCK_SIZE, 1, 1 );

void parallel_huffman_decode(
        const unsigned char* const h_input_buffer,
        unsigned int const& input_size,
        unsigned char* const h_output_buffer,
        unsigned int& output_size )
{
    // TODO: implement
}

typedef struct _code_word_t {
    unsigned int code;
    unsigned int code_size;
} code_word_t;

static code_word_t tree[UCHAR_MAX] = {0};

__global__ void init_positions(
        unsigned int* const d_positions,
        const unsigned int size )
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if ( idx < size ) {
        d_positions[idx] = idx;
    }
}

void parallel_huffman_encode(
        const unsigned char* const h_input_buffer,
        unsigned int const& input_size,
        unsigned char* const h_output_buffer,
        unsigned int& output_size )
{
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );
    const unsigned int stream_size = input_size / STREAM_COUNT;
    cudaStream_t streams[STREAM_COUNT];
    unsigned int j;

    for ( int i = 0 ; i < STREAM_COUNT ; ++i ) {
        checkCudaErrors( cudaStreamCreate( &streams[i] ) );
    }

    unsigned char* d_input;
    checkCudaErrors( cudaMalloc( &d_input, sizeof( *d_input ) * input_size ) );
    checkCudaErrors( cudaMemcpyAsync( d_input, h_input_buffer, input_size, cudaMemcpyHostToDevice, streams[0] ) );

    unsigned char* d_output;
    checkCudaErrors( cudaMalloc( &d_output, sizeof( *d_output ) * input_size ) );

    unsigned int* d_output_size;
    checkCudaErrors( cudaMalloc( &d_output_size, sizeof( *d_output_size ) ) );
    checkCudaErrors( cudaMemsetAsync( d_output_size, 0, sizeof( *d_output_size ), streams[1] ) );

    code_word_t* d_tree;
    const unsigned int tree_size = sizeof( tree );
    checkCudaErrors( cudaMalloc( &d_tree, tree_size ) );

    unsigned int* d_histogram;
    const unsigned int histogram_count = 1 << ( sizeof( *d_input ) << 3 );
    const unsigned int histogram_size = sizeof( *d_histogram ) << ( sizeof( *d_input ) << 3 );
    checkCudaErrors( cudaMalloc( &d_histogram, histogram_size ) );
    checkCudaErrors( cudaMemsetAsync( d_histogram, 0, histogram_size, streams[2] ) );

    unsigned int* d_sorted_histogram;
    checkCudaErrors( cudaMalloc( &d_sorted_histogram, histogram_size ) );

    unsigned int* d_relative_offsets;
    const unsigned int radix_relative_offsets_count = blelloch_size( histogram_count ) * RADIX_SORT_NUM_VALS;
    checkCudaErrors( cudaMalloc( &d_relative_offsets, radix_relative_offsets_count * sizeof( *d_relative_offsets ) ) );

    unsigned int* d_input_positions;
    checkCudaErrors( cudaMalloc( &d_input_positions, histogram_size ) );
    init_positions<<<grid_size, block_size, 0, streams[3]>>>( d_input_positions, histogram_count );

    unsigned int* d_output_positions;
    checkCudaErrors( cudaMalloc( &d_output_positions, histogram_size ) );

    unsigned int* d_radix_histogram;
    const unsigned int radix_blelloch_number = blelloch_size( RADIX_SORT_NUM_VALS );
    checkCudaErrors( cudaMalloc( &d_radix_histogram, radix_blelloch_number * sizeof( *d_radix_histogram ) ) );

    checkCudaErrors( cudaHostRegister( static_cast<void*>( const_cast<unsigned char*>( h_input_buffer ) ), input_size, cudaHostRegisterMapped ) );

    checkCudaErrors( cudaStreamSynchronize( streams[0] ) );

    histogram( d_histogram, histogram_count, d_input, input_size, streams[3] );

    checkCudaErrors( cudaDeviceSynchronize() );

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

    // TODO: generate huffman tree

    // TODO: map input symbols to output symbols and compact

    // TODO: output the huffman table along with output data

    // TODO: update reduce input size to actual compressed size

#if 0 
    checkCudaErrors( cudaMemcpy( &output_size, d_output_size, sizeof( *d_output_size ), cudaMemcpyDeviceToHost ) );
#else
    output_size = input_size;
#endif

    // TODO: update d_input with whatever buffer ends up being used

    checkCudaErrors( cudaHostUnregister( static_cast<void*>( const_cast<unsigned char*>( h_input_buffer ) ) ) );
    checkCudaErrors( cudaHostRegister( static_cast<void*>( h_output_buffer ), output_size, cudaHostRegisterMapped ) );

    j = 0;
    for ( unsigned int i = 0 ; i < output_size ; i += stream_size ) {
        const unsigned int chunk_size = ( i + stream_size > output_size ? output_size - i : stream_size );
        checkCudaErrors( cudaMemcpyAsync( &h_output_buffer[i], &d_input[i], chunk_size, cudaMemcpyDeviceToHost, streams[j] ) );
        j = ( j + 1 ) % STREAM_COUNT;
    }

    for ( int i = 0 ; i < STREAM_COUNT ; ++i ) {
        checkCudaErrors( cudaStreamSynchronize( streams[i] ) );
        checkCudaErrors( cudaStreamDestroy( streams[i] ) );
    }

    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaHostUnregister( static_cast<void*>( h_output_buffer ) ) );

    // TODO: release any additional memory allocated

    checkCudaErrors( cudaFree( d_input ) );
    checkCudaErrors( cudaFree( d_output ) );
    checkCudaErrors( cudaFree( d_output_size ) );
    checkCudaErrors( cudaFree( d_tree ) );
    checkCudaErrors( cudaFree( d_histogram ) );
    checkCudaErrors( cudaFree( d_sorted_histogram ) );
    checkCudaErrors( cudaFree( d_radix_histogram ) );
    checkCudaErrors( cudaFree( d_relative_offsets ) );
    checkCudaErrors( cudaFree( d_input_positions ) );
    checkCudaErrors( cudaFree( d_output_positions ) );
}
