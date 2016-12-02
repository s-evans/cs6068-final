#define RADIX_SORT_NUM_BITS 4 // number of bits used at a time for radix sort
#define RADIX_SORT_NUM_VALS ( 1 << RADIX_SORT_NUM_BITS ) // number of possible values given number of bits

void radix_sort(
        unsigned int* const d_input,
        unsigned int* const d_input_positions,
        unsigned int* const d_output,
        unsigned int* const d_output_positions,
        unsigned int const input_size,
        unsigned int* d_histogram,
        unsigned int* d_relative_offset,
        cudaStream_t stream );
