//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]

    In this assignment we will do 800 iterations.
   */

#include <stdio.h>
#include "utils.h"
#include <thrust/host_vector.h>
#include "limits.h"

#define BLOCK_DIM 128
#define NUM_BUFFERS 2

typedef enum _mask_t {
    INTERIOR = 0x0,
    WHITE    = 0x1,
    BORDER   = 0x2
} mask_t;

__device__ bool is_white( const uchar4* const pixel ) 
{
    return ( pixel->x / UCHAR_MAX ) & ( pixel->y / UCHAR_MAX ) & ( pixel->z / UCHAR_MAX );
}

__global__ void make_mask(
        const uchar4* const sourceImg,
        const size_t numRowsSource,
        const size_t numColsSource,
        unsigned char* const mask )
{
    // calculate image size
    const size_t img_size = numColsSource * numRowsSource;

    // calculate input data index
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( idx >= img_size ) {
        __syncthreads();
        return;
    }

    // allocate shared memory for each thread to access neighboring pixel
    __shared__ uchar4 cache[BLOCK_DIM];

    // shorten thread index
    const size_t tid = threadIdx.x;

    // populate shared memory
    cache[tid] = sourceImg[idx];

    // synchronize threads to ensure cache is fully populated
    __syncthreads();

    // determine whether or not the pixel is a border or interior pixel or is white
    if ( is_white( &cache[tid] ) ) {
        mask[idx] = WHITE;
    } else if ( is_white( tid != 0 ? &cache[tid - 1] : &sourceImg[idx % numColsSource != 0 ? idx - 1 : idx] ) ) { 
        mask[idx] = BORDER;
    } else if ( is_white( tid != BLOCK_DIM - 1 ? &cache[tid + 1] : &sourceImg[ ( idx + 1 ) % numColsSource != 0 ? ( idx + 1 ) : idx ] ) ) { 
        mask[idx] = BORDER;
    } else if ( is_white( &sourceImg[ idx > numColsSource ? idx - numColsSource : idx ] ) ) { 
        mask[idx] = BORDER;
    } else if ( is_white( &sourceImg[ idx < img_size - numColsSource ? idx + numColsSource : idx ] ) ) { 
        mask[idx] = BORDER;
    } else {
        mask[idx] = INTERIOR;
    }
}

__global__ void init_clone(
        const uchar4* const sourceImg,
        float3* const guess,
        const size_t img_size)
{
    // calculate input data index
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    // validate index
    if ( idx >= img_size ) {
        return;
    }

    // initialize and convert to float
    guess[idx].x = sourceImg[idx].x;
    guess[idx].y = sourceImg[idx].y;
    guess[idx].z = sourceImg[idx].z;
}

__global__ void end_clone(
        uchar4* const dst,
        const float3* const src,
        const size_t img_size)
{
    // calculate input data index
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    // validate index
    if ( idx >= img_size ) {
        return;
    }

    // copy and convert from float to uchar
    dst[idx].x = src[idx].x;
    dst[idx].y = src[idx].y;
    dst[idx].z = src[idx].z;
}

__global__ void do_clone(
        const uchar4* const sourceImg,
        const uchar4* const destImg,
        const unsigned char* const mask,
        const float3* const prev,
        float3* const next,
        const size_t numRowsSource,
        const size_t numColsSource)
{
    // calculate image size
    const size_t img_size = numColsSource * numRowsSource;

    // calculate input data index
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t idx_left = ( idx % numColsSource != 0 ? idx - 1 : idx );
    const size_t idx_right = ( ( idx + 1 ) % numColsSource != 0 ? ( idx + 1 ) : idx );
    const size_t idx_above = ( idx > numColsSource ? idx - numColsSource : idx );
    const size_t idx_below = ( idx < img_size - numColsSource ? idx + numColsSource : idx );

    // validate index
    if ( idx >= img_size ) {
        return;
    }

    if ( mask[idx] == INTERIOR ) {
        // for each channel

        {
            float sum1 = 0.0f;
            float sum2 = 0.0f;

            // for each neighbor 

            {
                const size_t neighbor_idx = idx_left;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].x;
                } else {
                    sum1 += destImg[neighbor_idx].x;
                }
                sum2 += sourceImg[idx].x - sourceImg[neighbor_idx].x;
            }

            {
                const size_t neighbor_idx = idx_right;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].x;
                } else {
                    sum1 += destImg[neighbor_idx].x;
                }
                sum2 += sourceImg[idx].x - sourceImg[neighbor_idx].x;
            }

            {
                const size_t neighbor_idx = idx_above;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].x;
                } else {
                    sum1 += destImg[neighbor_idx].x;
                }
                sum2 += sourceImg[idx].x - sourceImg[neighbor_idx].x;
            }

            {
                const size_t neighbor_idx = idx_below;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].x;
                } else {
                    sum1 += destImg[neighbor_idx].x;
                }
                sum2 += sourceImg[idx].x - sourceImg[neighbor_idx].x;
            }

            const float newVal = ( sum1 + sum2 ) / 4.0f;
            next[idx].x = min( 255.0f, max( 0.0f, newVal ) ); // clamp to [0, 255]
        }

        {
            float sum1 = 0.0f;
            float sum2 = 0.0f;

            // for each neighbor 

            {
                const size_t neighbor_idx = idx_left;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].y;
                } else {
                    sum1 += destImg[neighbor_idx].y;
                }
                sum2 += sourceImg[idx].y - sourceImg[neighbor_idx].y;
            }

            {
                const size_t neighbor_idx = idx_right;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].y;
                } else {
                    sum1 += destImg[neighbor_idx].y;
                }
                sum2 += sourceImg[idx].y - sourceImg[neighbor_idx].y;
            }

            {
                const size_t neighbor_idx = idx_above;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].y;
                } else {
                    sum1 += destImg[neighbor_idx].y;
                }
                sum2 += sourceImg[idx].y - sourceImg[neighbor_idx].y;
            }

            {
                const size_t neighbor_idx = idx_below;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].y;
                } else {
                    sum1 += destImg[neighbor_idx].y;
                }
                sum2 += sourceImg[idx].y - sourceImg[neighbor_idx].y;
            }

            const float newVal = ( sum1 + sum2 ) / 4.0f;
            next[idx].y = min( 255.0f, max( 0.0f, newVal ) ); // clamp to [0, 255]
        }

        {
            float sum1 = 0.0f;
            float sum2 = 0.0f;

            // for each neighbor 

            {
                const size_t neighbor_idx = idx_left;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].z;
                } else {
                    sum1 += destImg[neighbor_idx].z;
                }
                sum2 += sourceImg[idx].z - sourceImg[neighbor_idx].z;
            }

            {
                const size_t neighbor_idx = idx_right;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].z;
                } else {
                    sum1 += destImg[neighbor_idx].z;
                }
                sum2 += sourceImg[idx].z - sourceImg[neighbor_idx].z;
            }

            {
                const size_t neighbor_idx = idx_above;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].z;
                } else {
                    sum1 += destImg[neighbor_idx].z;
                }
                sum2 += sourceImg[idx].z - sourceImg[neighbor_idx].z;
            }

            {
                const size_t neighbor_idx = idx_below;
                if ( mask[neighbor_idx] == INTERIOR ) {
                    sum1 += prev[neighbor_idx].z;
                } else {
                    sum1 += destImg[neighbor_idx].z;
                }
                sum2 += sourceImg[idx].z - sourceImg[neighbor_idx].z;
            }

            const float newVal = ( sum1 + sum2 ) / 4.0f;
            next[idx].z = min( 255.0f, max( 0.0f, newVal ) ); // clamp to [0, 255]
        }

    } else {
        next[idx].x = destImg[idx].x;
        next[idx].y = destImg[idx].y;
        next[idx].z = destImg[idx].z;
    }
}

void your_blend(
        const uchar4* const h_sourceImg,  //IN
        const size_t numRowsSource,
        const size_t numColsSource,
        const uchar4* const h_destImg, //IN
        uchar4* const h_blendedImg) //OUT
{
    const size_t img_size = numRowsSource * numColsSource;
    const size_t img_size_bytes = img_size * sizeof( uchar4 );
    const dim3 block_size( BLOCK_DIM, 1, 1 );
    const dim3 grid_size( ( img_size + block_size.x - 1 ) / block_size.x, 1, 1 );

    /*
       1) Compute a mask of the pixels from the source image to be copied
       The pixels that shouldn't be copied are completely white, they
       have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
       */

    /*
       2) Compute the interior and border regions of the mask.  An interior
       pixel has all 4 neighbors also inside the mask.  A border pixel is
       in the mask itself, but has at least one neighbor that isn't.
       */

    unsigned char* d_mask;
    checkCudaErrors( cudaMalloc( &d_mask, img_size * sizeof( unsigned char ) ) );

    uchar4* d_sourceImg;
    checkCudaErrors( cudaMalloc( &d_sourceImg, img_size_bytes ) );

    uchar4* d_destImg;
    checkCudaErrors( cudaMalloc( &d_destImg, img_size_bytes ) );

    float3* d_blendedImg_a;
    checkCudaErrors( cudaMalloc( &d_blendedImg_a, img_size * sizeof( float3 ) ) );

    float3* d_blendedImg_b;
    checkCudaErrors( cudaMalloc( &d_blendedImg_b, img_size * sizeof( float3 ) ) );

    uchar4* d_blendedImg_c;
    checkCudaErrors( cudaMalloc( &d_blendedImg_c, img_size_bytes ) );

    checkCudaErrors( cudaHostRegister( (void*) h_sourceImg, img_size_bytes, 0 ) );
    checkCudaErrors( cudaHostRegister( (void*) h_destImg, img_size_bytes, 0 ) );
    checkCudaErrors( cudaHostRegister( (void*) h_blendedImg, img_size_bytes, 0 ) );

    cudaStream_t stream;
    checkCudaErrors( cudaStreamCreate( &stream ) );

    cudaStream_t stream2;
    checkCudaErrors( cudaStreamCreate( &stream2 ) );

    checkCudaErrors( cudaMemcpyAsync( d_sourceImg, h_sourceImg, img_size_bytes, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpyAsync( d_destImg, h_destImg, img_size_bytes, cudaMemcpyHostToDevice, stream ) );
    init_clone<<<grid_size, block_size, 0, stream2>>>( d_sourceImg, d_blendedImg_a, img_size );

    make_mask<<<grid_size, block_size>>>( d_sourceImg, numRowsSource, numColsSource, d_mask );
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaStreamDestroy( stream ) );
    checkCudaErrors( cudaStreamDestroy( stream2 ) );

    /*
       3) Separate out the incoming image into three separate channels
       */

    /*
       4) Create two float(!) buffers for each color channel that will
       act as our guesses.  Initialize them to the respective color
       channel of the source image since that will act as our initial guess.
       */

    /*
       5) For each color channel perform the Jacobi iteration described
       above 800 times.
       */

    /*
       6) Create the output image by replacing all the interior pixels
       in the destination image with the result of the Jacobi iterations.
       Just cast the floating point values to unsigned chars since we have
       already made sure to clamp them to the correct range.
       */

    unsigned int i = 0;

    while ( i < 800 ) {
        float3* const prev = ( i % 2 ? d_blendedImg_b : d_blendedImg_a );
        float3* const next = ( i % 2 ? d_blendedImg_a : d_blendedImg_b );
        do_clone<<<grid_size, block_size>>>( d_sourceImg, d_destImg, d_mask, prev, next, numRowsSource, numColsSource );
        checkCudaErrors( cudaDeviceSynchronize() );
        ++i;
    }

    end_clone<<<grid_size, block_size>>>( d_blendedImg_c, ( i % 2 ? d_blendedImg_b : d_blendedImg_a ), img_size_bytes );
    checkCudaErrors( cudaDeviceSynchronize() );

    checkCudaErrors( cudaMemcpy( h_blendedImg, d_blendedImg_c, img_size_bytes, cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaHostUnregister( (void*) h_sourceImg ) );
    checkCudaErrors( cudaHostUnregister( (void*) h_destImg ) );
    checkCudaErrors( cudaHostUnregister( (void*) h_blendedImg ) );

    checkCudaErrors( cudaFree( d_sourceImg ) );
    checkCudaErrors( cudaFree( d_mask ) );
    checkCudaErrors( cudaFree( d_destImg ) );
    checkCudaErrors( cudaFree( d_blendedImg_a ) );
    checkCudaErrors( cudaFree( d_blendedImg_b ) );
}
