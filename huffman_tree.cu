#include "huffman_tree.h"
#include "blelloch_scan.h"
#include "utils.h"

#include <ostream>

typedef struct _code_word_t {
    unsigned int code[8];
    unsigned int size;
} code_word_t;

struct node_t {
    unsigned int weight;
    unsigned char left_idx;
    unsigned char symbol;
};

__device__ unsigned int d_start_idx;

__global__ void find_start_idx(
        const unsigned int* const sorted_histogram )
{
    const unsigned int idx = threadIdx.x;

    if ( sorted_histogram[idx] == 0 ) {
        atomicAdd( &d_start_idx, 1 );
    }
}

__global__ void initialize_nodes(
        const unsigned int* const sorted_histogram,
        const unsigned int* const sorted_symbols,
        node_t* const nodes,
        const unsigned int populated_nodes
        )
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

__global__ void insert_super_nodes (
        node_t* const pnodes )
{
    __shared__ unsigned int not_moved;
    __shared__ unsigned int left_weight;
    __shared__ unsigned int right_weight;
    __shared__ unsigned int new_weight;
    const unsigned int idx = threadIdx.x;
    const node_t* const end_nodes = pnodes + blockDim.x;

    // TODO: update to use shared memory?

    for ( node_t* nodes = pnodes ; nodes < end_nodes ; nodes += 2 ) {
        const unsigned int offset = nodes - pnodes;

        if ( idx == 0 ) {
            not_moved = 0;
            left_weight = nodes[0].weight;
            right_weight = nodes[1].weight;
            new_weight = left_weight + right_weight;
        }

        __syncthreads();

        if ( left_weight == 0 || right_weight == 0 ) {
            return;
        }

        const unsigned int my_weight = nodes[idx].weight;

        if ( my_weight == 0 ) {
            return;
        }

        const bool move = ( my_weight >= new_weight );
        node_t node;

        if ( move ) {
            node = nodes[idx];
        } else {
            atomicAdd( &not_moved, 1 );
        }

        __syncthreads();

        if ( move ) {
            nodes[idx + 1] = node;
        } else if ( idx == 0 ) {
            node_t* const pnode = &nodes[not_moved];
            pnode->left_idx = offset / 2;
            pnode->symbol = 0;
            pnode->weight = new_weight;
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
    // TODO: use shared memory?

    const unsigned int tid = threadIdx.x;
    const unsigned int direction = tid % 2;
    unsigned short stack[1024];

    if ( !nodes[tid].weight || !nodes[tid + 1].weight ) {
        return;
    }

    init( stack );
    push( stack, tid, 0 );

    while ( !empty( stack ) ) {

        unsigned short idx;
        unsigned short bit_offset;

        pop( stack, &idx, &bit_offset );

        if ( nodes[idx].left_idx == UCHAR_MAX ) {
            const unsigned char symbol = nodes[idx].symbol;
            atomicOr( &code_word_map[symbol].code[ bit_offset / 32 ], direction << ( bit_offset % 32 ) );
            atomicAdd( &code_word_map[symbol].size, 1 );
            continue;
        }

        idx = nodes[idx].left_idx * 2;
        ++bit_offset;
        push( stack, idx, bit_offset );
        push( stack, idx + 1, bit_offset );
    }
}

__global__ void generate_output(
        unsigned char* const output,
        unsigned char const* const input,
        unsigned int const input_size,
        const code_word_t* const code_word_map )
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int idx = tid;
    const unsigned int symbol = input[idx];

    code_word_map[symbol].code;
    code_word_map[symbol].size;

    // TODO: implement
}

void huffman_encode(
        unsigned char* const d_output,
        unsigned int* const output_size,
        unsigned char const* const d_input,
        unsigned int const input_size,
        unsigned int const* const d_sorted_histogram,
        unsigned int const* const d_sorted_symbols )
{
    const unsigned int histogram_size = 256;

    // TODO: rename file

    code_word_t* d_code_word_map;
    const unsigned int code_map_size = sizeof( *d_code_word_map ) * 256;
    checkCudaErrors( cudaMalloc( &d_code_word_map, code_map_size ) );
    checkCudaErrors( cudaMemsetAsync( d_code_word_map, 0, code_map_size, 0 ) );

    void* p_d_start_idx;
    checkCudaErrors( cudaGetSymbolAddress( &p_d_start_idx, d_start_idx ) );
    checkCudaErrors( cudaMemsetAsync( p_d_start_idx, 0, sizeof( d_start_idx ), 0 ) );

    find_start_idx<<<1, histogram_size, 0, 0>>>( d_sorted_histogram );

    unsigned int start_idx;
    checkCudaErrors( cudaMemcpyAsync( &start_idx, p_d_start_idx, sizeof( start_idx ), cudaMemcpyDeviceToHost, 0 ) );

    checkCudaErrors( cudaStreamSynchronize( 0 ) );

    const unsigned int node_count = histogram_size - start_idx;
    const unsigned int max_node_count = ( blelloch_size( node_count ) << 1 );

    node_t* d_nodes;
    checkCudaErrors( cudaMalloc( &d_nodes, sizeof( *d_nodes ) * max_node_count ) );
    initialize_nodes<<<1, max_node_count, 0, 0>>>(
            &d_sorted_histogram[start_idx],
            &d_sorted_symbols[start_idx],
            d_nodes,
            node_count );

    /* std::cerr << "node_count: " << node_count << std::endl; */
    /* std::cerr << "max_node_count: " << max_node_count << std::endl; */

    insert_super_nodes<<<1, max_node_count, 0, 0>>>( d_nodes );

    checkCudaErrors( cudaStreamSynchronize( 0 ) );
    for ( unsigned int i = 0 ; i < max_node_count ; ++i ) {
        node_t tmp;
        checkCudaErrors( cudaMemcpy( &tmp, &d_nodes[i], sizeof( tmp ), cudaMemcpyDeviceToHost ) );
        std::cerr << "idx: " << i << "; " << tmp << std::endl;
    }

    generate_code_words<<<1, max_node_count, 0, 0>>>( d_code_word_map, d_nodes );

    checkCudaErrors( cudaStreamSynchronize( 0 ) );
    for ( unsigned int i = 0 ; i < 256 ; ++i ) {
        code_word_t tmp;
        checkCudaErrors( cudaMemcpy( &tmp, &d_code_word_map[i], sizeof( tmp ), cudaMemcpyDeviceToHost ) );
        std::cerr << "idx: " << i << "; " << tmp << std::endl;
    }

    /* std::cerr << "code_map_size: " << code_map_size << "; " << std::endl; */

    // TODO: generate output string of code words

    const dim3 block_size( 256, 1, 1 );
    const dim3 grid_size( ( input_size + block_size.x - 1 ) / block_size.x, 1, 1 );
    generate_output<<<grid_size, block_size, 0, 0>>>(
            d_output,
            d_input,
            input_size,
            d_code_word_map );

    // TODO: ensure that the new output size gets set
    // TODO: copy output buffer from device to host
    // TODO: must output the code word table as well

    checkCudaErrors( cudaFree( d_code_word_map ) );
    checkCudaErrors( cudaFree( d_nodes ) );
}
