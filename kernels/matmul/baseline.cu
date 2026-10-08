#include <cuda_runtime.h>

#define TILE_SIZE 16

// KernelForge baseline: tiled FP32 matrix multiplication.
//
// A: M x K
// B: K x N
// C: M x N
//
// Launch configuration:
//   Threads per block: (TILE_SIZE, TILE_SIZE)
//   Blocks per grid:   (ceil(N / TILE_SIZE), ceil(M / TILE_SIZE))

extern "C" __global__ void kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
  // Shared-memory tiles reused by threads in the block.
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = blockIdx.y * TILE_SIZE + ty;
  const int col = blockIdx.x * TILE_SIZE + tx;

  float acc = 0.0f;

  // Number of tiles required to cover the reduction dimension K.
  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t)
  {
    const int a_col = t * TILE_SIZE + tx;
    const int b_row = t * TILE_SIZE + ty;

    // Cooperatively load a tile of A.
    if (row < M && a_col < K)
    {
      tile_A[ty][tx] = A[row * K + a_col];
    }
    else
    {
      tile_A[ty][tx] = 0.0f;
    }

    // Cooperatively load a tile of B.
    if (b_row < K && col < N)
    {
      tile_B[ty][tx] = B[b_row * N + col];
    }
    else
    {
      tile_B[ty][tx] = 0.0f;
    }

    // Ensure all tile values are ready before computation.
    __syncthreads();

// Accumulate this tile's contribution to C[row, col].
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k)
    {
      acc += tile_A[ty][k] * tile_B[k][tx];
    }

    // Ensure no thread overwrites shared memory too early.
    __syncthreads();
  }

  // Store the result if this thread maps to a valid output element.
  if (row < M && col < N)
  {
    C[row * N + col] = acc;
  }
}
