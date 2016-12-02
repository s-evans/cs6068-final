void relative_offsets(
        unsigned int* const d_relative_offsets,
        const unsigned int* const d_input_values,
        const unsigned int input_size,
        const unsigned int predicate_value,
        const unsigned int mask_offset,
        const unsigned int mask,
        cudaStream_t stream );
