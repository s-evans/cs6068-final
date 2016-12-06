#include "huffman_tree.h"
#include "limits.h"
#include "parallel_huffman.h"
#include "utils.h"

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

// NOTE: function taken from thrust histogram example

template <typename Vector1, typename Vector2>
void dense_histogram(const Vector1& input, Vector2& histogram )
{
    typedef typename Vector1::value_type ValueType; // input value type
    typedef typename Vector2::value_type IndexType; // histogram index type

    // copy input data (could be skipped if input is allowed to be modified)
    thrust::device_vector<ValueType> data( input );

    // sort data to bring equal elements together
    thrust::sort( data.begin(), data.end() );

    // number of histogram bins is equal to the maximum value plus one
    IndexType num_bins = data.back() + 1;

    // resize histogram storage
    histogram.resize( num_bins );

    // find the end of each bin of values
    thrust::counting_iterator<IndexType> search_begin( 0 );
    thrust::upper_bound(
            data.begin(), data.end(),
            search_begin, search_begin + num_bins,
            histogram.begin() );

    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(
            histogram.begin(), histogram.end(), histogram.begin() );
}

void parallel_huffman_encode(
        const unsigned char* const h_input_buffer,
        unsigned int const input_size,
        unsigned char* const h_output_buffer,
        unsigned int& output_size )
{
    cudaStream_t s;
    checkCudaErrors( cudaStreamCreate( &s ) );

    cudaStream_t t;
    checkCudaErrors( cudaStreamCreate( &t ) );

    thrust::device_vector<unsigned char> d_input( input_size );
    checkCudaErrors( cudaMemcpyAsync( thrust::raw_pointer_cast( &d_input[0] ), h_input_buffer, input_size, cudaMemcpyHostToDevice, s ) );

    thrust::device_vector<unsigned char> d_output( input_size );
    thrust::fill( thrust::cuda::par( t ), d_output.begin(), d_output.end(), 0 );

    thrust::device_vector<unsigned int> d_histogram( 256 );
    dense_histogram( d_input, d_histogram );

    thrust::device_vector<unsigned int> d_input_positions( 256 );
    thrust::sequence( thrust::cuda::par( s ), d_input_positions.begin(), d_input_positions.end() );

    thrust::sort_by_key( d_histogram.begin(), d_histogram.end(), d_input_positions.begin() );

    huffman_encode(
            d_output,
            &output_size,
            d_input,
            d_histogram,
            d_input_positions );

    checkCudaErrors( cudaMemcpy( h_output_buffer, thrust::raw_pointer_cast( &d_output[0] ), output_size, cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaStreamDestroy( s ) );
    checkCudaErrors( cudaStreamDestroy( t ) );
}
