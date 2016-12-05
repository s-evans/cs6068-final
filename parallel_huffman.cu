#include "debug_print.h"
#include "histogram.h"
#include "huffman_tree.h"
#include "limits.h"
#include "parallel_huffman.h"
#include "utils.h"

#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void parallel_huffman_encode(
        const unsigned char* const h_input_buffer,
        unsigned int const input_size,
        unsigned char* const h_output_buffer,
        unsigned int& output_size )
{
    cudaStream_t s;
    checkCudaErrors( cudaStreamCreate( &s ) );

    // TODO: use streams for memcpy and histogram generation
    thrust::device_vector<unsigned char> d_input( input_size );
    checkCudaErrors( cudaHostRegister( static_cast<void*>( const_cast<unsigned char*>( h_input_buffer ) ), input_size, cudaHostRegisterMapped ) );
    checkCudaErrors( cudaMemcpyAsync( thrust::raw_pointer_cast( &d_input[0] ), h_input_buffer, input_size, cudaMemcpyHostToDevice, s ) );

    /* std::cerr << std::endl << "input file" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < input_size ; i += 16 ) { */
    /*     unsigned char tmp[16] = {0}; */
    /*     const unsigned int remain = input_size - i; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_input[i] ), ( remain / 16 ? 16 : remain ), cudaMemcpyDeviceToHost ) ); */
    /*     fprintf( stderr, "%08x: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n", */ 
    /*                 i, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15] ); */
    /* } */

    // TODO: use a stream
    thrust::device_vector<unsigned char> d_output( input_size );
    thrust::fill( d_output.begin(), d_output.end(), 0 );

    /* std::cerr << std::endl << "zeroed output file" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < input_size ; i += 16 ) { */
    /*     unsigned char tmp[16] = {0}; */
    /*     const unsigned int remain = input_size - i; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_output[i] ), ( remain / 16 ? 16 : remain ), cudaMemcpyDeviceToHost ) ); */
    /*     fprintf( stderr, "%08x: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n", */ 
    /*                 i, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15] ); */
    /* } */

    const unsigned int histogram_count = 256;

    // TODO: use a stream
    thrust::device_vector<unsigned int> d_histogram( 256 );
    thrust::fill( d_histogram.begin(), d_histogram.end(), 0 );

    /* std::cerr << std::endl << "zeroed histogram" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < 256 ; ++i ) { */
    /*     unsigned int tmp; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_histogram[i] ), sizeof( tmp ), cudaMemcpyDeviceToHost ) ); */
    /*     std::cerr << "idx: " << i << "; " << tmp << std::endl; */
    /* } */

    // TODO: use a stream
    thrust::device_vector<unsigned int> d_input_positions( 256 );
    thrust::sequence( d_input_positions.begin(), d_input_positions.end() );

    /* std::cerr << std::endl << "initial positions" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < 256 ; ++i ) { */
    /*     unsigned int tmp; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_input_positions[i] ), sizeof( tmp ), cudaMemcpyDeviceToHost ) ); */
    /*     std::cerr << "idx: " << i << "; " << tmp << std::endl; */
    /* } */

    checkCudaErrors( cudaStreamSynchronize( s ) );

    histogram( thrust::raw_pointer_cast( &d_histogram[0] ), histogram_count, thrust::raw_pointer_cast( &d_input[0] ), input_size );

    /* std::cerr << std::endl << "histogram" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < 256 ; ++i ) { */
    /*     unsigned int tmp; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_histogram[i] ), sizeof( tmp ), cudaMemcpyDeviceToHost ) ); */
    /*     std::cerr << "idx: " << i << "; " << tmp << std::endl; */
    /* } */

    thrust::sort_by_key( d_histogram.begin(), d_histogram.end(), d_input_positions.begin() );

    /* std::cerr << std::endl << "sorted histogram" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < 256 ; ++i ) { */
    /*     unsigned int tmp; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_histogram[i] ), sizeof( tmp ), cudaMemcpyDeviceToHost ) ); */
    /*     std::cerr << "idx: " << i << "; " << tmp << std::endl; */
    /* } */

    /* std::cerr << std::endl << "sorted positions" << std::endl; */
    /* checkCudaErrors( cudaStreamSynchronize( 0 ) ); */
    /* for ( unsigned int i = 0 ; i < 256 ; ++i ) { */
    /*     unsigned int tmp; */
    /*     checkCudaErrors( cudaMemcpy( &tmp, thrust::raw_pointer_cast( &d_input_positions[i] ), sizeof( tmp ), cudaMemcpyDeviceToHost ) ); */
    /*     std::cerr << "idx: " << i << "; " << tmp << std::endl; */
    /* } */

    /* checkCudaErrors( cudaDeviceSynchronize() ); */

    /* std::cerr << "sorted histogram" << std::endl; */
    /* debug_print( d_sorted_histogram, histogram_count ); */

    /* std::cerr << "sorted symbols" << std::endl; */
    /* debug_print( d_output_positions, histogram_count ); */

    huffman_encode(
            d_output,
            &output_size,
            d_input,
            d_histogram,
            d_input_positions );

    checkCudaErrors( cudaMemcpy( h_output_buffer, thrust::raw_pointer_cast( &d_output[0] ), output_size, cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaStreamDestroy( s ) );
    checkCudaErrors( cudaHostUnregister( static_cast<void*>( const_cast<unsigned char*>( h_input_buffer ) ) ) );
}
