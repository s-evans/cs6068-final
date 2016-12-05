#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <thrust/device_vector.h>

void huffman_encode(
        thrust::device_vector<unsigned char>& d_output, 
        unsigned int* const output_size,
        thrust::device_vector<unsigned char>& d_input, 
        thrust::device_vector<unsigned int>& d_sorted_histogram,
        thrust::device_vector<unsigned int>& d_sorted_symbols );


#endif // HUFFMAN_TREE_H
