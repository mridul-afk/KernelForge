#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <cstring>

// ------------------------------------------------------------
// KernelForge Benchmark Configuration
// ------------------------------------------------------------

constexpr int TILE_SIZE = 16;
constexpr int WARMUP_ITERATIONS = 10;
constexpr int BENCHMARK_ITERATIONS = 100;

constexpr float ABS_TOLERANCE = 1e-3f;
constexpr float REL_TOLERANCE = 1e-3f;

// Matrix multiplication kernel defined in kernels/matmul/baseline.cu.
// Keep this signature consistent with the kernel definition.
extern "C" __global__ void kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K);

// ------------------------------------------------------------
// CUDA Error Checking
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                             \
  do                                                                 \
  {                                                                  \
    cudaError_t error = (call);                                      \
    if (error != cudaSuccess)                                        \
    {                                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                << "\n  " << cudaGetErrorString(error) << std::endl; \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                \
  } while (0)

// ------------------------------------------------------------
// Deterministic Matrix Initialization
// ------------------------------------------------------------

void initMatrix(std::vector<float> &matrix, int rows, int cols)
{
  for (int i = 0; i < rows * cols; ++i)
  {
    // Repeatable, nonuniform values in a small range.
    matrix[i] = static_cast<float>((i * 17 + 13) % 101 - 50) / 50.0f;
  }
}

// ------------------------------------------------------------
// CPU Reference Matrix Multiplication
// A: M x K
// B: K x N
// C: M x N
// ------------------------------------------------------------

void cpuMatMul(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K)
{
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float sum = 0.0f;

      for (int k = 0; k < K; ++k)
      {
        sum += A[i * K + k] * B[k * N + j];
      }

      C[i * N + j] = sum;
    }
  }
}

// ------------------------------------------------------------
// Correctness Validation
// ------------------------------------------------------------

struct ValidationResult
{
  bool passed = true;
  std::size_t mismatches = 0;
  float maxAbsoluteError = 0.0f;
  std::size_t firstMismatch = std::numeric_limits<std::size_t>::max();
};

ValidationResult validateResults(
    const std::vector<float> &actual,
    const std::vector<float> &expected)
{
  ValidationResult result;

  for (std::size_t i = 0; i < actual.size(); ++i)
  {
    const float a = actual[i];
    const float b = expected[i];

    if (!std::isfinite(a) || !std::isfinite(b))
    {
      result.passed = false;
      ++result.mismatches;

      if (result.firstMismatch ==
          std::numeric_limits<std::size_t>::max())
      {
        result.firstMismatch = i;
      }

      continue;
    }

    const float absError = std::fabs(a - b);
    result.maxAbsoluteError =
        std::max(result.maxAbsoluteError, absError);

    const float allowedError =
        ABS_TOLERANCE + REL_TOLERANCE * std::fabs(b);

    if (absError > allowedError)
    {
      result.passed = false;
      ++result.mismatches;

      if (result.firstMismatch ==
          std::numeric_limits<std::size_t>::max())
      {
        result.firstMismatch = i;
      }
    }
  }

  return result;
}

// ------------------------------------------------------------
// Main Benchmark
// ------------------------------------------------------------

int main(int argc, char *argv[])
{
  // Initial rectangular test:
  // A = M x K, B = K x N, C = M x N
  int M = 512;
  int K = 1024;
  int N = 512;

  // Optional command-line dimensions: M N K
  if (argc == 4)
  {
    try
    {
      M = std::atoi(argv[1]);
      N = std::atoi(argv[2]);
      K = std::atoi(argv[3]);
    }
    catch (...)
    {
      std::cerr << "Usage: runner.exe [M N K]\n";
      return EXIT_FAILURE;
    }
  }
  else if (argc != 1)
  {
    std::cerr << "Usage: runner.exe [M N K]\n";
    return EXIT_FAILURE;
  }

  if (M <= 0 || N <= 0 || K <= 0)
  {
    std::cerr << "Matrix dimensions must be positive.\n";
    return EXIT_FAILURE;
  }

  std::cout << "========================================\n";
  std::cout << "         KernelForge Benchmark\n";
  std::cout << "========================================\n";
  std::cout << "Matrix A: " << M << " x " << K << '\n';
  std::cout << "Matrix B: " << K << " x " << N << '\n';
  std::cout << "Matrix C: " << M << " x " << N << '\n';

  // --------------------------------------------------------
  // 1. Initialize host matrices
  // --------------------------------------------------------

  const std::size_t countA =
      static_cast<std::size_t>(M) * K;
  const std::size_t countB =
      static_cast<std::size_t>(K) * N;
  const std::size_t countC =
      static_cast<std::size_t>(M) * N;

  const std::size_t bytesA = countA * sizeof(float);
  const std::size_t bytesB = countB * sizeof(float);
  const std::size_t bytesC = countC * sizeof(float);

  std::vector<float> h_A(countA);
  std::vector<float> h_B(countB);
  std::vector<float> h_C(countC);
  std::vector<float> h_reference(countC);

  initMatrix(h_A, M, K);
  initMatrix(h_B, K, N);

  // --------------------------------------------------------
  // 2. Compute trusted CPU reference
  // --------------------------------------------------------

  std::cout << "\nComputing CPU reference...\n";

  cpuMatMul(h_A, h_B, h_reference, M, N, K);

  // --------------------------------------------------------
  // 3. Allocate GPU memory
  // --------------------------------------------------------

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), bytesA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), bytesB));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), bytesC));

  // Copy inputs to GPU. Transfer time is not benchmarked.
  CUDA_CHECK(cudaMemcpy(
      d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(
      d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

  // --------------------------------------------------------
  // 4. Configure and launch kernel
  // --------------------------------------------------------

  const dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

  const dim3 blocksPerGrid(
      (N + TILE_SIZE - 1) / TILE_SIZE,
      (M + TILE_SIZE - 1) / TILE_SIZE);

  auto launchKernel = [&]()
  {
    kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, M, N, K);

    // Detect launch/configuration errors.
    CUDA_CHECK(cudaGetLastError());
  };

  // --------------------------------------------------------
  // 5. Initial execution and correctness check
  // --------------------------------------------------------

  std::cout << "\nRunning correctness test...\n";

  launchKernel();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(
      h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

  const ValidationResult validation =
      validateResults(h_C, h_reference);

  std::cout << "Correctness: "
            << (validation.passed ? "PASS" : "FAIL")
            << '\n';

  std::cout << "Maximum absolute error: "
            << std::scientific << validation.maxAbsoluteError
            << '\n';

  std::cout << "Mismatched elements: "
            << validation.mismatches << '\n';

  if (!validation.passed)
  {
    const std::size_t i = validation.firstMismatch;

    std::cerr << "First mismatch at flat index " << i
              << ": GPU = " << h_C[i]
              << ", CPU = " << h_reference[i] << '\n';

    // Do not benchmark an incorrect kernel.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_FAILURE;
  }

  std::cout << "Sample C[0,0]: " << h_C[0] << '\n';

  // --------------------------------------------------------
  // 6. Warm up the GPU
  // --------------------------------------------------------

  std::cout << "\nWarming up GPU...\n";

  for (int i = 0; i < WARMUP_ITERATIONS; ++i)
  {
    launchKernel();
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // --------------------------------------------------------
  // 7. Benchmark using CUDA events
  // --------------------------------------------------------

  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  std::cout << "Benchmarking kernel...\n";

  CUDA_CHECK(cudaEventRecord(startEvent));

  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
  {
    launchKernel();
  }

  CUDA_CHECK(cudaEventRecord(stopEvent));
  CUDA_CHECK(cudaEventSynchronize(stopEvent));

  float elapsedMs = 0.0f;

  CUDA_CHECK(cudaEventElapsedTime(
      &elapsedMs, startEvent, stopEvent));

  const float averageLatencyMs =
      elapsedMs / BENCHMARK_ITERATIONS;

  // --------------------------------------------------------
  // 8. Report benchmark results
  // --------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << "           Benchmark Results\n";
  std::cout << "========================================\n";

  std::cout << "Kernel: tiled FP32 matrix multiplication\n";
  std::cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << '\n';
  std::cout << "Correctness: PASS\n";
  std::cout << "Warm-up iterations: " << WARMUP_ITERATIONS << '\n';
  std::cout << "Timed iterations: " << BENCHMARK_ITERATIONS << '\n';

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Total timed GPU execution: "
            << elapsedMs << " ms\n";

  std::cout << "Average kernel latency: "
            << averageLatencyMs << " ms\n";

  // --------------------------------------------------------
  // 9. Cleanup
  // --------------------------------------------------------

  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  std::cout << "\nBenchmark completed successfully.\n";

  return EXIT_SUCCESS;
}
