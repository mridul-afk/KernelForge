#include <cuda_runtime.h>

// KernelForge baseline: tiled FP32 matrix multiplication.
//
//   A: M x K   (row-major)
//   B: K x N   (row-major)
//   C: M x N   (row-major)
//
// KERNEL CONTRACT
// ---------------
// A candidate kernel file must export exactly one host entry point:
//
//   extern "C" cudaError_t kernelforge_launch(
//       const float *A, const float *B, float *C,
//       int M, int N, int K, cudaStream_t stream);
//
// The file owns EVERYTHING about how the kernel runs: tile size, block shape,
// grid shape, shared memory size, thread coarsening. The benchmark runner knows
// nothing about them, so any of them can change freely without touching the
// runner. The entry point must launch asynchronously on `stream` and return
// cudaGetLastError().

#define TILE_SIZE 16

__global__ void matmul_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
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
      tile_A[ty][tx] = A[(size_t)row * K + a_col];
    }
    else
    {
      tile_A[ty][tx] = 0.0f;
    }

    // Cooperatively load a tile of B.
    if (b_row < K && col < N)
    {
      tile_B[ty][tx] = B[(size_t)b_row * N + col];
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
    C[(size_t)row * N + col] = acc;
  }
}

extern "C" cudaError_t kernelforge_launch(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
  const dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  const dim3 blocksPerGrid(
      (N + TILE_SIZE - 1) / TILE_SIZE,
      (M + TILE_SIZE - 1) / TILE_SIZE);

  matmul_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      A, B, C, M, N, K);

  return cudaGetLastError();
}
