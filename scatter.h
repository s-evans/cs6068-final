void scatter(
        unsigned int* const d_histogram,
        const unsigned int histogram_size,
        const unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int mask_offset,
        const unsigned int mask,
        cudaStream_t stream );
