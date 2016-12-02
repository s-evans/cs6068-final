unsigned int blelloch_size ( unsigned int n );

void blelloch_scan(
        unsigned int* const d_input,
        const unsigned int input_size,
        cudaStream_t stream );
