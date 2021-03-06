#include "huffman_tree.h"
#include "utils.h"
#include <stdio.h>

#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <ostream>

unsigned int blelloch_size ( unsigned int n )
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return ++n;
}

typedef struct _code_word_t {
    unsigned int code[8];
    unsigned int size;
} code_word_t;

struct node_t {
    unsigned int weight;
    unsigned char left_idx;
    unsigned char symbol;
};

__global__ void initialize_nodes(
        const unsigned int* const sorted_histogram,
        const unsigned int* const sorted_symbols,
        node_t* const nodes,
        const unsigned int populated_nodes )
{
    const unsigned int idx = threadIdx.x;

    if ( idx < populated_nodes ) {
        nodes[idx].left_idx = UCHAR_MAX;
        nodes[idx].symbol = static_cast<short>( sorted_symbols[idx] );
        nodes[idx].weight = sorted_histogram[idx];
    } else {
        nodes[idx].left_idx = UCHAR_MAX;
        nodes[idx].symbol = 0;
        nodes[idx].weight = 0;
    }
}

std::ostream& operator<< ( std::ostream& stream, node_t const& node )
{
    return stream << "node.left_idx: " << static_cast<int>( node.left_idx == UCHAR_MAX ? -1 : node.left_idx * 2 )
        << "; node.symbol: "  << static_cast<unsigned int>( node.symbol )
        << "; node.weight: " << static_cast<unsigned int>( node.weight )
        << ";";
}

std::ostream& operator<< ( std::ostream& stream, code_word_t const& code_word )
{
    stream << "; code_word.size: "  << static_cast<unsigned int>( code_word.size ) << "; code word: ";

    for ( int i = code_word.size - 1 ; i >= 0 ; --i ) {
        stream << static_cast<bool>( ( code_word.code[i / 32] >> ( i % 32 ) ) & 1 );
    }

    return stream << ";";
}

__global__ void insert_super_nodes(
    node_t* const pnodes )
{
    __shared__ unsigned int not_moved;
    __shared__ unsigned int new_weight;
    __shared__ node_t snodes[512];

    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    snodes[idx] = pnodes[idx];

    __syncthreads();

    for ( unsigned int offset = 0 ; offset < blockDim.x ; offset += 2 ) {

        if ( idx == 0 ) {
            not_moved = 0;
            new_weight = snodes[offset].weight + snodes[offset + 1].weight;
        }

        const unsigned int my_weight = snodes[offset + idx].weight;

        __syncthreads();

        if ( my_weight == 0 ) {
            return;
        }

        if ( my_weight >= new_weight ) {
            const node_t node = snodes[offset + idx];
            __syncthreads();
            snodes[offset + idx + 1] = node;
        } else {
            atomicAdd( &not_moved, 1 );
            __syncthreads();
        }

        if ( idx == 0 && not_moved ) {
            node_t* const pnode = &snodes[offset + not_moved];
            pnode->left_idx = offset / 2;
            pnode->symbol = 0;
            pnode->weight = new_weight;
        }

        if ( idx == 0 || idx == 1 ) {
            pnodes[offset + idx] = snodes[offset + idx];
        }

        __syncthreads();
    }
}

__device__ void init( unsigned short* const stack )
{
    stack[0] = 1;
}

__device__ void push(
    unsigned short* const stack,
    unsigned short const idx,
    unsigned short const bit_offset )
{
    const unsigned short ptr = stack[0];
    stack[0] += 2;
    stack[ptr] = idx;
    stack[ptr + 1] = bit_offset;
}

__device__ void pop(
    unsigned short* const stack,
    unsigned short* const idx,
    unsigned short* const bit_offset )
{
    const unsigned short ptr = stack[0] - 2;
    stack[0] -= 2;
    *idx = stack[ptr];
    *bit_offset = stack[ptr + 1];
}

__device__ bool empty( const unsigned short* const stack )
{
    return ( stack[0] == 1 );
}

__global__ void generate_code_words(
    code_word_t* const code_word_map,
    const node_t* const nodes )
{
    const unsigned int tid = threadIdx.x;
    const unsigned int direction = tid % 2;
    unsigned short stack[2 * ( 511 + 1 )];

    if ( !nodes[tid].weight || !nodes[tid + 1].weight ) {
        return;
    }

    __shared__ node_t s_nodes[512];
    s_nodes[tid] = nodes[tid];
    __syncthreads();

    init( stack );
    push( stack, tid, 0 );

    while ( !empty( stack ) ) {

        unsigned short idx;
        unsigned short bit_offset;

        pop( stack, &idx, &bit_offset );

        if ( s_nodes[idx].left_idx == UCHAR_MAX ) {
            const unsigned char symbol = s_nodes[idx].symbol;
            atomicOr( &code_word_map[symbol].code[ bit_offset / 32 ], direction << ( bit_offset % 32 ) );
            atomicAdd( &code_word_map[symbol].size, 1 );
            continue;
        }

        idx = s_nodes[idx].left_idx * 2;
        ++bit_offset;

        push( stack, idx, bit_offset );
        push( stack, idx + 1, bit_offset );
    }
}

__global__ void map_output_positions(
    unsigned int* const output_positions,
    unsigned char const* const input,
    unsigned int const input_size,
    const code_word_t* const code_word_map )
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if ( tid >= input_size ) {
        return;
    }

    const unsigned int symbol = input[tid];
    const unsigned int size = code_word_map[symbol].size;
    output_positions[tid] = size;
}

__global__ void generate_output(
    unsigned char* const output,
    unsigned char const* const input,
    unsigned int const input_size,
    code_word_t const* const code_word_map,
    unsigned int const* const output_positions )
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if ( tid >= input_size ) {
        return;
    }

    __shared__ code_word_t s_code_word_map[256];

    if ( threadIdx.x < 256 ) { 
        s_code_word_map[threadIdx.x] = code_word_map[threadIdx.x];
    }

    __syncthreads();

    const unsigned int bit_offset = output_positions[tid];
    unsigned int* const output_words = reinterpret_cast<unsigned int*>( output );

    const unsigned int symbol = input[tid];
    const code_word_t* const code_word = &s_code_word_map[symbol];
    const unsigned int codeword_bits = code_word->size;

    unsigned char stage[4] = {0};

    for ( unsigned int i = 0 ; i < codeword_bits ; ++i ) {

        const unsigned int code_word_bit_position = codeword_bits - i - 1;
        const unsigned int code_word_word_position = code_word_bit_position / 32;
        const unsigned int code_word_word_bit_position = code_word_bit_position % 32;

        const unsigned int bit = ( code_word->code[code_word_word_position] >> code_word_word_bit_position ) & 1;

        const unsigned int current_bit_offset = bit_offset + i;
        const unsigned int word_bit_offset = current_bit_offset % 32;
        const unsigned int word_byte_offset = word_bit_offset / 8;
        const unsigned int word_byte_bit_offset = 7 - ( current_bit_offset % 8 );

        stage[word_byte_offset] |= bit << word_byte_bit_offset;

        if ( i == codeword_bits - 1 || word_bit_offset == 31 ) {
            const unsigned int word_offset = current_bit_offset / 32;
            atomicOr( output_words + word_offset, *reinterpret_cast<unsigned int*>( &stage ) );
            stage[3] = stage[2] = stage[1] = stage[0] = 0;
        }
    }
}

void huffman_encode(
        thrust::device_vector<unsigned char>& d_output, 
        unsigned int* const output_size,
        thrust::device_vector<unsigned char>& d_input, 
        thrust::device_vector<unsigned int>& d_sorted_histogram,
        thrust::device_vector<unsigned int>& d_sorted_symbols )
{
    const unsigned int input_size = d_input.size();

    thrust::device_vector<code_word_t> d_code_word_map( d_sorted_histogram.size() );
    checkCudaErrors( cudaMemsetAsync(
                thrust::raw_pointer_cast( &d_code_word_map[0] ),
                0, 
                sizeof( code_word_t ) * d_code_word_map.size(),
                0 ) );

    unsigned int start_idx = thrust::count( d_sorted_histogram.begin(), d_sorted_histogram.end(), 0 );

    const unsigned int node_count = d_sorted_histogram.size() - start_idx;
    const unsigned int max_node_count = ( blelloch_size( node_count ) << 1 );

    thrust::device_vector<node_t> d_nodes( max_node_count );
    initialize_nodes<<< 1, max_node_count, 0, 0>>>(
        thrust::raw_pointer_cast( &d_sorted_histogram[start_idx] ),
        thrust::raw_pointer_cast( &d_sorted_symbols[start_idx] ),
        thrust::raw_pointer_cast( &d_nodes[0] ),
        node_count );

    insert_super_nodes<<< 1, max_node_count, 0, 0>>>( thrust::raw_pointer_cast( &d_nodes[0] ) );

    generate_code_words<<< 1, max_node_count, 0, 0>>>( thrust::raw_pointer_cast( &d_code_word_map[0] ), thrust::raw_pointer_cast( &d_nodes[0] ) );

    const dim3 block_size( 256, 1, 1 );
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );

    const unsigned int output_positions_count = input_size + 1;
    thrust::device_vector<unsigned int> d_output_positions( output_positions_count );

    map_output_positions<<<grid_size, block_size, 0, 0>>>(
        thrust::raw_pointer_cast( &d_output_positions[0] ),
        thrust::raw_pointer_cast( &d_input[0] ),
        input_size,
        thrust::raw_pointer_cast( &d_code_word_map[0] ) );

    thrust::exclusive_scan( d_output_positions.begin(), d_output_positions.end(), d_output_positions.begin() );

    generate_output<<<grid_size, block_size, 0, 0>>>(
            thrust::raw_pointer_cast( &d_output[0] ),
            thrust::raw_pointer_cast( &d_input[0] ),
            input_size,
            thrust::raw_pointer_cast( &d_code_word_map[0] ),
            thrust::raw_pointer_cast( &d_output_positions[0] ) );

    checkCudaErrors( cudaMemcpy( output_size, thrust::raw_pointer_cast( &d_output_positions[0] ) + input_size, sizeof( *output_size ), cudaMemcpyDeviceToHost ) );
    *output_size = ( *output_size + 8 - 1 ) / 8;
}
