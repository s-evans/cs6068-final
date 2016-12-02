#include "debug_print.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

void debug_print( const unsigned int* const device_ptr, const unsigned int size )
{
    unsigned int* ptr = (unsigned int*) malloc( size * sizeof( unsigned int ) );
    checkCudaErrors( cudaMemcpy( ptr, device_ptr, size * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
    std::cerr << std::endl;
    for ( unsigned int j = 0 ; j < size ; ++j ) {
        std::cerr << ptr[j] << std::endl;
    }
    std::cerr << std::endl;
    free( ptr );
}
