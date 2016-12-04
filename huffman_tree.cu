#include "huffman_tree.h"
#include "blelloch_scan.h"
#include "utils.h"

#include <ostream>

typedef struct _code_word_t {
    unsigned int code;
    unsigned int size;
} code_word_t;

struct node_t {
    unsigned int weight;
    unsigned short left_idx;
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
        nodes[idx].left_idx = USHRT_MAX;
        nodes[idx].symbol = static_cast<short>( sorted_symbols[idx] );
        nodes[idx].weight = sorted_histogram[idx];
    } else {
        nodes[idx].left_idx = USHRT_MAX;
        nodes[idx].symbol = 0;
        nodes[idx].weight = 0;
    }

}

std::ostream& operator<< ( std::ostream& stream, node_t const& node )
{
    return stream << "node.left_idx: " << static_cast<unsigned int>( node.left_idx ) 
        << "; node.symbol: "  << static_cast<unsigned int>( node.symbol ) 
        << "; node.weight: " << static_cast<unsigned int>( node.weight ) 
        << ";";
}

std::ostream& operator<< ( std::ostream& stream, code_word_t const& code_word )
{
    stream << "code_word.code: " << static_cast<unsigned int>( code_word.code )
        << "; code_word.size: "  << static_cast<unsigned int>( code_word.size )
        << "; code word: "; 

    for ( unsigned int i = 0 ; i < code_word.size ; ++i ) {
        stream << static_cast<bool>( ( code_word.code >> ( code_word.size - i - 1 ) ) & 1 );
    }

    return stream << ";";
}

__global__ void insert_super_node (
        node_t* const nodes,
        unsigned int offset )
{
    __shared__ unsigned int not_moved;
    __shared__ unsigned int left_weight;
    __shared__ unsigned int right_weight;
    __shared__ unsigned int new_weight;
    const unsigned int idx = threadIdx.x;

    if ( idx == 0 ) {
        not_moved = 0;
        left_weight = nodes[0].weight;
        right_weight = nodes[1].weight;
        new_weight = left_weight + right_weight;
    }

    __syncthreads();

    if ( left_weight == 0 || right_weight == 0 ) {
        __syncthreads();
        return;
    }

    const unsigned int my_weight = nodes[idx].weight;

    if ( my_weight == 0 ) {
        __syncthreads();
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
        pnode->left_idx = offset;
        pnode->symbol = 0;
        pnode->weight = new_weight;
    }
}

__global__ void generate_code_words(
    code_word_t* code_word_map,
    const node_t* const nodes )
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int init_idx = tid / 2;
    const unsigned int thread_dir = tid % 2;

    // TODO: fix this

    /* for ( unsigned int depth_limit = 1 ; depth_limit < 3 ; ++depth_limit  ) { */
    for ( unsigned int depth_limit = 1 ; depth_limit < blockDim.x / 2 ; ++depth_limit  ) {

        unsigned int idx = init_idx;

        for ( unsigned int depth = 0 ; depth < depth_limit ; ++depth ) {

            if ( idx == USHRT_MAX ) {
                continue;
            }

            if ( nodes[idx].weight == 0 ) {
                idx = USHRT_MAX;
                continue;
            }

            if ( nodes[idx].left_idx == USHRT_MAX ) {
                idx = USHRT_MAX;
                continue;
            }

            idx = nodes[idx].left_idx + thread_dir;
        }

        if ( idx == USHRT_MAX ) {
            continue;
        }

        if ( nodes[idx].left_idx != USHRT_MAX ) {
            continue;
        }

        const unsigned char symbol = nodes[idx].symbol;
        atomicOr( &code_word_map[symbol].code, thread_dir << ( depth_limit - 1 ) );
        atomicAdd( &code_word_map[symbol].size, 1 );
    }
}

void make_huffman_tree(
        const unsigned int* const d_sorted_histogram,
        const unsigned int* const d_sorted_symbols)
{
    const unsigned int histogram_size = 256;

    // TODO: free device buffer eventually
    // TODO: do something useful with the tree

    // TODO: rename function
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
    const unsigned int max_node_count = ( blelloch_size( node_count ) << 1 ) - 1;

    node_t* d_nodes;
    checkCudaErrors( cudaMalloc( &d_nodes, sizeof( *d_nodes ) * max_node_count ) );
    initialize_nodes<<<1, max_node_count, 0, 0>>>(
            &d_sorted_histogram[start_idx], 
            &d_sorted_symbols[start_idx],
            d_nodes,
            node_count );

    // TODO: move this loop to the kernel

    for ( unsigned int i = 0 ; i < max_node_count ; i += 2 ) {
        insert_super_node<<<1, max_node_count - i, 0, 0>>>( &d_nodes[i], i );
    }

    generate_code_words<<<1, max_node_count * 2, 0, 0>>>( d_code_word_map, d_nodes );

    /* std::cerr << "node_count: " << node_count << std::endl; */
    /* std::cerr << "max_node_count: " << max_node_count << std::endl; */

    checkCudaErrors( cudaStreamSynchronize( 0 ) );
    for ( unsigned int i = 0 ; i < max_node_count ; ++i ) {
        node_t tmp;
        checkCudaErrors( cudaMemcpy( &tmp, &d_nodes[i], sizeof( tmp ), cudaMemcpyDeviceToHost ) );
        std::cerr << "idx: " << i << "; " << tmp << std::endl;
    }

    checkCudaErrors( cudaStreamSynchronize( 0 ) );
    for ( unsigned int i = 0 ; i < UCHAR_MAX ; ++i ) {
        code_word_t tmp;
        checkCudaErrors( cudaMemcpy( &tmp, &d_code_word_map[i], sizeof( tmp ), cudaMemcpyDeviceToHost ) );
        std::cerr << "idx: " << i << "; " << tmp << std::endl;
    }
}
