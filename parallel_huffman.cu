#include "parallel_huffman.h"
#include "utils.h"

// TODO: kernels for creating a histogram, sorting, mapping codes, and reducing (concatenating) codes

#define STREAM_SIZE 32768
#define STREAM_COUNT 3

void parallel_huffman_decode(
        const char* const input_buffer,
        unsigned int const& input_size,
        char* const output_buffer,
        unsigned int& output_size )
{
    // TODO: implement
}

// TODO: define static tree
static const unsigned char tree[] = {0};

void parallel_huffman_encode(
        const char* const input_buffer,
        unsigned int const& input_size,
        char* const output_buffer,
        unsigned int& output_size )
{
    cudaStream_t streams[STREAM_COUNT];

    for ( int i = 0 ; i < STREAM_COUNT ; ++i ) {
        checkCudaErrors( cudaStreamCreate( &streams[i] ) );
    }

    // TODO: allocate device memory for histogram buffer for sorting

    char* d_input;
    checkCudaErrors( cudaMalloc( &d_input, input_size ) );

    char* d_output;
    checkCudaErrors( cudaMalloc( &d_output, input_size ) );

    // TODO: update size and type for the tree

    unsigned int* d_output_size;
    checkCudaErrors( cudaMalloc( &d_output_size, sizeof( *d_output_size ) ) );
    checkCudaErrors( cudaMemsetAsync( d_output_size, 0, sizeof( *d_output_size ), streams[0] ) );

    char* d_tree;
    const unsigned int tree_size = sizeof( tree );
    checkCudaErrors( cudaMalloc( &d_tree, tree_size ) );
    checkCudaErrors( cudaMemcpyAsync( d_tree, tree, tree_size, cudaMemcpyHostToDevice, streams[1] ) );

    char* d_histo;
    const unsigned int histo_size = 256 * sizeof( unsigned int );
    checkCudaErrors( cudaMalloc( &d_histo, histo_size ) );
    checkCudaErrors( cudaMemsetAsync( d_histo, 0, histo_size, streams[2] ) );

    // TODO: ensure there are no races from memset'ing and memcpy'ing

    // TODO: may want to do a dynamic number of streams?
    // TODO: may want to do a dynamic stream chunk size?

    checkCudaErrors( cudaHostRegister( static_cast<void*>( const_cast<char*>( input_buffer ) ), input_size, cudaHostRegisterMapped ) );

    unsigned int j = 0;

    for ( unsigned int i = 0 ; i < input_size ; i += STREAM_SIZE ) {
        const unsigned int chunk_size = ( i + STREAM_SIZE > input_size ? input_size - i : STREAM_SIZE );
        checkCudaErrors( cudaMemcpyAsync( &d_input[i], &input_buffer[i], chunk_size, cudaMemcpyHostToDevice, streams[j] ) );
        j = ( j + 1 ) % STREAM_COUNT;
    }

    // TODO: histogram kernel

    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaHostUnregister( static_cast<void*>( const_cast<char*>( input_buffer ) ) ) );

    // TODO: sort

    // TODO: move huffman tree elements with sorted values

    // TODO: map input symbols to output symbols and compact

    // TODO: output the huffman table along with output data

    // TODO: update reduce input size to actual compressed size

#if 0 
    checkCudaErrors( cudaMemcpy( &output_size, d_output_size, sizeof( *d_output_size ), cudaMemcpyDeviceToHost ) );
#else
    output_size = input_size;
#endif

    j = 0;

    // TODO: update d_input with whatever buffer ends up being used

    checkCudaErrors( cudaHostRegister( static_cast<void*>( output_buffer ), output_size, cudaHostRegisterMapped ) );

    for ( unsigned int i = 0 ; i < output_size ; i += STREAM_SIZE ) {
        const unsigned int chunk_size = ( i + STREAM_SIZE > output_size ? output_size - i : STREAM_SIZE );
        checkCudaErrors( cudaMemcpyAsync( &output_buffer[i], &d_input[i], chunk_size, cudaMemcpyDeviceToHost, streams[j] ) );
        j = ( j + 1 ) % STREAM_COUNT;
    }

    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaHostUnregister( static_cast<void*>( output_buffer ) ) );

    // TODO: release any additional memory allocated

    checkCudaErrors( cudaFree( d_input ) );
    checkCudaErrors( cudaFree( d_output ) );
    checkCudaErrors( cudaFree( d_output_size ) );
    checkCudaErrors( cudaFree( d_tree ) );
    checkCudaErrors( cudaFree( d_histo ) );

    for ( int i = 0 ; i < STREAM_COUNT ; ++i ) {
        checkCudaErrors( cudaStreamDestroy( streams[i] ) );
    }
}
