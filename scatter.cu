#include "blelloch_scan.h"
#include "limits.h"
#include "masked_histogram.h"
#include "scatter.h"
#include "utils.h"

void scatter(
        unsigned int* const d_histogram,
        const unsigned int histogram_size,
        const unsigned int* const d_input,
        const unsigned int input_size,
        const unsigned int mask_offset,
        const unsigned int mask,
        cudaStream_t stream )
{
    checkCudaErrors( cudaMemsetAsync( d_histogram, 0, histogram_size * sizeof( *d_histogram ), stream ) );

    masked_histogram( d_histogram, histogram_size, d_input, input_size, mask_offset, mask, stream );

    blelloch_scan( d_histogram, histogram_size, stream );
}
