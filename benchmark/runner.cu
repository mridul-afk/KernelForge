// ----------------------------------------------------------------------------
// KernelForge benchmark runner
//
//   runner [M N K]        performance shape (default 2048 2048 2048)
//
// stdout : exactly one JSON object (machine-readable, consumed by the agent)
// stderr : human-readable progress log
//
// Exit codes: 0 = ok, 1 = usage error, 2 = incorrect result, 3 = runtime error
//
// The candidate kernel file must export `kernelforge_launch` (see
// kernels/matmul/baseline.cu). The runner knows nothing about tile sizes,
// block shapes or grid shapes.
// ----------------------------------------------------------------------------

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" cudaError_t kernelforge_launch(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// ------------------------------------------------------------
// Configuration
// ------------------------------------------------------------

namespace
{
  constexpr int WARMUP_ITERATIONS = 10;
  constexpr int TIMING_REPEATS = 15;
  constexpr int ITERATIONS_PER_REPEAT = 20;

  // allowed |gpu - ref| = ABS_TOL_SCALE * sqrt(K) + REL_TOL * |ref|
  constexpr double ABS_TOL_SCALE = 1e-4;
  constexpr double REL_TOL = 1e-3;

  constexpr int DEFAULT_PERF_DIM = 2048;

  // Correctness shapes (M, N, K). Deliberately includes sizes that are NOT
  // multiples of any common tile size, plus degenerate and skinny shapes.
  const std::vector<std::array<int, 3>> CORRECTNESS_SHAPES = {
      {1, 1, 1},
      {16, 16, 16},
      {33, 17, 129},
      {64, 64, 64},
      {257, 255, 31},
      {513, 1000, 77},
      {512, 512, 1024},
  };

  std::string g_stage = "startup";
}

// ------------------------------------------------------------
// Error handling: throw instead of exit so main() can always emit JSON.
// ------------------------------------------------------------

struct KernelForgeError : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

#define CUDA_CHECK(call)                                           \
  do                                                               \
  {                                                                \
    cudaError_t error_ = (call);                                   \
    if (error_ != cudaSuccess)                                     \
    {                                                              \
      std::ostringstream os_;                                      \
      os_ << cudaGetErrorString(error_) << " (" << __FILE__ << ":" \
          << __LINE__ << ")";                                      \
      throw KernelForgeError(os_.str());                           \
    }                                                              \
  } while (0)

#define CUBLAS_CHECK(call)                                             \
  do                                                                   \
  {                                                                    \
    cublasStatus_t status_ = (call);                                   \
    if (status_ != CUBLAS_STATUS_SUCCESS)                              \
    {                                                                  \
      std::ostringstream os_;                                          \
      os_ << "cuBLAS error code " << static_cast<int>(status_) << " (" \
          << __FILE__ << ":" << __LINE__ << ")";                       \
      throw KernelForgeError(os_.str());                               \
    }                                                                  \
  } while (0)

// ------------------------------------------------------------
// Small RAII wrappers
// ------------------------------------------------------------

struct DeviceBuffer
{
  float *ptr = nullptr;
  size_t count = 0;

  explicit DeviceBuffer(size_t n) : count(n)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(float)));
  }
  ~DeviceBuffer()
  {
    if (ptr)
      cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void upload(const std::vector<float> &host)
  {
    CUDA_CHECK(cudaMemcpy(ptr, host.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  void download(std::vector<float> &host) const
  {
    CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  // 0xFF bytes == NaN as float. A kernel that fails to write an output
  // element can never pass by accident.
  void poison()
  {
    CUDA_CHECK(cudaMemset(ptr, 0xFF, count * sizeof(float)));
  }
};

struct CudaEvent
{
  cudaEvent_t event = nullptr;
  CudaEvent() { CUDA_CHECK(cudaEventCreate(&event)); }
  ~CudaEvent()
  {
    if (event)
      cudaEventDestroy(event);
  }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
};

struct CublasHandle
{
  cublasHandle_t handle = nullptr;
  CublasHandle()
  {
    CUBLAS_CHECK(cublasCreate(&handle));
    // Strict FP32 so the baseline is comparable to a plain FP32 kernel.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
  ~CublasHandle()
  {
    if (handle)
      cublasDestroy(handle);
  }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
};

// ------------------------------------------------------------
// JSON helpers
// ------------------------------------------------------------

std::string jnum(double v)
{
  if (!std::isfinite(v))
    return "null";
  std::ostringstream os;
  os << std::setprecision(9) << v;
  return os.str();
}

std::string jstr(const std::string &s)
{
  std::ostringstream os;
  os << '"';
  for (unsigned char c : s)
  {
    switch (c)
    {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (c < 0x20)
        os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
           << static_cast<int>(c) << std::dec << std::setfill(' ');
      else
        os << c;
    }
  }
  os << '"';
  return os.str();
}

const char *jbool(bool b) { return b ? "true" : "false"; }

// ------------------------------------------------------------
// Data generation and CPU reference
// ------------------------------------------------------------

// Seeded, platform-independent values in [-1, 1).
void fillRandom(std::vector<float> &v, uint32_t seed)
{
  uint32_t s = seed * 2654435761u + 12345u;
  for (float &x : v)
  {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>(s >> 8) / 8388608.0f - 1.0f;
  }
}

// Double-precision accumulation: the trusted reference.
std::vector<double> cpuMatMul(
    const std::vector<float> &A,
    const std::vector<float> &B,
    int M,
    int N,
    int K)
{
  std::vector<double> C(static_cast<size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i)
  {
    for (int k = 0; k < K; ++k)
    {
      const double a = A[static_cast<size_t>(i) * K + k];
      for (int j = 0; j < N; ++j)
      {
        C[static_cast<size_t>(i) * N + j] +=
            a * B[static_cast<size_t>(k) * N + j];
      }
    }
  }
  return C;
}

// ------------------------------------------------------------
// Validation
// ------------------------------------------------------------

struct Validation
{
  bool passed = true;
  size_t mismatches = 0;
  double maxAbsError = 0.0;
  bool hasFirst = false;
  size_t firstIndex = 0;
  double firstActual = 0.0;
  double firstExpected = 0.0;
};

template <typename T>
Validation validate(
    const std::vector<float> &actual,
    const std::vector<T> &expected,
    int K)
{
  Validation r;
  const double absTol = ABS_TOL_SCALE * std::sqrt(static_cast<double>(K));

  for (size_t i = 0; i < actual.size(); ++i)
  {
    const double a = actual[i];
    const double e = expected[i];
    bool bad = false;

    if (!std::isfinite(a) || !std::isfinite(e))
    {
      bad = true;
    }
    else
    {
      const double err = std::fabs(a - e);
      r.maxAbsError = std::max(r.maxAbsError, err);
      bad = err > absTol + REL_TOL * std::fabs(e);
    }

    if (bad)
    {
      r.passed = false;
      ++r.mismatches;
      if (!r.hasFirst)
      {
        r.hasFirst = true;
        r.firstIndex = i;
        r.firstActual = a;
        r.firstExpected = e;
      }
    }
  }
  return r;
}

std::string validationJson(const Validation &v, int N)
{
  std::ostringstream os;
  os << "\"passed\":" << jbool(v.passed)
     << ",\"max_abs_error\":" << jnum(v.maxAbsError)
     << ",\"mismatches\":" << v.mismatches;
  if (v.hasFirst)
  {
    os << ",\"first_mismatch\":{\"row\":" << v.firstIndex / N
       << ",\"col\":" << v.firstIndex % N
       << ",\"gpu\":" << jnum(v.firstActual)
       << ",\"expected\":" << jnum(v.firstExpected) << "}";
  }
  return os.str();
}

// ------------------------------------------------------------
// Correctness test for one shape
// ------------------------------------------------------------

Validation runCorrectnessCase(int M, int N, int K, uint32_t seed)
{
  std::ostringstream stage;
  stage << "correctness " << M << "x" << N << "x" << K;
  g_stage = stage.str();

  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N);
  fillRandom(hA, seed);
  fillRandom(hB, seed + 7919u);

  const std::vector<double> reference = cpuMatMul(hA, hB, M, N, K);

  DeviceBuffer dA(hA.size()), dB(hB.size()), dC(hC.size());
  dA.upload(hA);
  dB.upload(hB);
  dC.poison();

  CUDA_CHECK(kernelforge_launch(dA.ptr, dB.ptr, dC.ptr, M, N, K, nullptr));
  // Surfaces asynchronous failures (illegal address, etc.) here.
  CUDA_CHECK(cudaDeviceSynchronize());

  dC.download(hC);
  return validate(hC, reference, K);
}

// ------------------------------------------------------------
// Timing
// ------------------------------------------------------------

struct TimingStats
{
  double minMs = 0, medianMs = 0, meanMs = 0, stddevMs = 0;
};

template <typename Launch>
TimingStats timeKernel(Launch &&launch)
{
  for (int i = 0; i < WARMUP_ITERATIONS; ++i)
    launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaEvent start, stop;
  std::vector<double> samples;
  samples.reserve(TIMING_REPEATS);

  for (int r = 0; r < TIMING_REPEATS; ++r)
  {
    CUDA_CHECK(cudaEventRecord(start.event));
    for (int i = 0; i < ITERATIONS_PER_REPEAT; ++i)
      launch();
    CUDA_CHECK(cudaEventRecord(stop.event));
    CUDA_CHECK(cudaEventSynchronize(stop.event));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.event, stop.event));
    samples.push_back(static_cast<double>(ms) / ITERATIONS_PER_REPEAT);
  }

  std::sort(samples.begin(), samples.end());

  TimingStats s;
  s.minMs = samples.front();
  s.medianMs = samples[samples.size() / 2];
  double sum = 0;
  for (double v : samples)
    sum += v;
  s.meanMs = sum / samples.size();
  double var = 0;
  for (double v : samples)
    var += (v - s.meanMs) * (v - s.meanMs);
  s.stddevMs = std::sqrt(var / samples.size());
  return s;
}

std::string timingJson(const TimingStats &t, double flops)
{
  std::ostringstream os;
  os << "{\"median_ms\":" << jnum(t.medianMs)
     << ",\"min_ms\":" << jnum(t.minMs)
     << ",\"mean_ms\":" << jnum(t.meanMs)
     << ",\"stddev_ms\":" << jnum(t.stddevMs)
     << ",\"gflops\":" << jnum(flops / (t.medianMs * 1e6)) << "}";
  return os.str();
}

// ------------------------------------------------------------
// Main benchmark
// ------------------------------------------------------------

int runBenchmark(int M, int N, int K)
{
  g_stage = "device query";
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::cerr << "KernelForge benchmark on " << prop.name << " (sm_"
            << prop.major << prop.minor << ")\n";

  std::ostringstream deviceJson;
  deviceJson << "{\"name\":" << jstr(prop.name) << ",\"compute_capability\":\""
             << prop.major << "." << prop.minor << "\"}";

  // ---- 1. Correctness over many shapes (do not stop at the first failure;
  //         the agent benefits from seeing every failing shape). ----
  bool allPassed = true;
  std::ostringstream casesJson;
  casesJson << "[";
  uint32_t seed = 1;
  for (size_t idx = 0; idx < CORRECTNESS_SHAPES.size(); ++idx)
  {
    const auto &s = CORRECTNESS_SHAPES[idx];
    const Validation v = runCorrectnessCase(s[0], s[1], s[2], seed++);
    std::cerr << "  correctness " << s[0] << "x" << s[1] << "x" << s[2]
              << ": " << (v.passed ? "PASS" : "FAIL") << "\n";
    allPassed = allPassed && v.passed;

    if (idx)
      casesJson << ",";
    casesJson << "{\"M\":" << s[0] << ",\"N\":" << s[1] << ",\"K\":" << s[2]
              << "," << validationJson(v, s[1]) << "}";
  }
  casesJson << "]";

  if (!allPassed)
  {
    std::cout << "{\"status\":\"incorrect\",\"device\":" << deviceJson.str()
              << ",\"correctness\":{\"passed\":false,\"cases\":"
              << casesJson.str() << "}}\n";
    return 2;
  }

  // ---- 2. Performance shape: set up data and the cuBLAS reference. ----
  g_stage = "performance setup";
  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N);
  std::vector<float> hRef(static_cast<size_t>(M) * N);
  fillRandom(hA, 1001u);
  fillRandom(hB, 2002u);

  DeviceBuffer dA(hA.size()), dB(hB.size()), dC(hC.size()), dRef(hRef.size());
  dA.upload(hA);
  dB.upload(hB);

  CublasHandle blas;
  const float alpha = 1.0f, beta = 0.0f;

  // Row-major C = A*B  <=>  column-major C^T = B^T * A^T.
  auto cublasLaunch = [&]()
  {
    CUBLAS_CHECK(cublasSgemm(
        blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha, dB.ptr, N, dA.ptr, K,
        &beta, dRef.ptr, N));
  };
  auto kernelLaunch = [&]()
  {
    CUDA_CHECK(kernelforge_launch(dA.ptr, dB.ptr, dC.ptr, M, N, K, nullptr));
  };

  g_stage = "cuBLAS reference";
  cublasLaunch();
  CUDA_CHECK(cudaDeviceSynchronize());
  dRef.download(hRef);

  g_stage = "performance-shape correctness";
  dC.poison();
  kernelLaunch();
  CUDA_CHECK(cudaDeviceSynchronize());
  dC.download(hC);
  const Validation preCheck = validate(hC, hRef, K);
  std::cerr << "  perf-shape correctness vs cuBLAS: "
            << (preCheck.passed ? "PASS" : "FAIL") << "\n";

  // ---- 3. Timing. ----
  const double flops = 2.0 * M * static_cast<double>(N) * K;

  g_stage = "timing kernel";
  const TimingStats kernelStats = timeKernel(kernelLaunch);

  // Re-check after the timed loop: catches races / nondeterminism that
  // happened to produce correct output on a single run.
  g_stage = "post-benchmark correctness";
  CUDA_CHECK(cudaDeviceSynchronize());
  dC.download(hC);
  const Validation postCheck = validate(hC, hRef, K);
  std::cerr << "  post-benchmark correctness: "
            << (postCheck.passed ? "PASS" : "FAIL") << "\n";

  g_stage = "timing cuBLAS";
  const TimingStats blasStats = timeKernel(cublasLaunch);

  const bool ok = preCheck.passed && postCheck.passed;
  const double percentOfCublas = 100.0 * blasStats.medianMs / kernelStats.medianMs;

  std::cerr << std::fixed << std::setprecision(4)
            << "  kernel median " << kernelStats.medianMs << " ms, cuBLAS median "
            << blasStats.medianMs << " ms, " << std::setprecision(1)
            << percentOfCublas << "% of cuBLAS\n";

  std::cout << "{\"status\":" << (ok ? "\"ok\"" : "\"incorrect\"")
            << ",\"device\":" << deviceJson.str()
            << ",\"correctness\":{\"passed\":" << jbool(allPassed)
            << ",\"cases\":" << casesJson.str() << "}"
            << ",\"perf_shape\":{\"M\":" << M << ",\"N\":" << N
            << ",\"K\":" << K << "}"
            << ",\"perf_shape_check\":{" << validationJson(preCheck, N) << "}"
            << ",\"post_benchmark_check\":{" << validationJson(postCheck, N) << "}"
            << ",\"kernel\":" << timingJson(kernelStats, flops)
            << ",\"cublas\":" << timingJson(blasStats, flops)
            << ",\"percent_of_cublas\":" << jnum(percentOfCublas)
            << "}\n";

  return ok ? 0 : 2;
}

// ------------------------------------------------------------
// Entry point
// ------------------------------------------------------------

bool parseDim(const char *text, int &out)
{
  char *end = nullptr;
  const long v = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || v <= 0 || v > 65536)
    return false;
  out = static_cast<int>(v);
  return true;
}

int main(int argc, char *argv[])
{
  int M = DEFAULT_PERF_DIM, N = DEFAULT_PERF_DIM, K = DEFAULT_PERF_DIM;

  if (argc == 4)
  {
    if (!parseDim(argv[1], M) || !parseDim(argv[2], N) || !parseDim(argv[3], K))
    {
      std::cerr << "Dimensions must be integers in [1, 65536].\n";
      return 1;
    }
  }
  else if (argc != 1)
  {
    std::cerr << "Usage: runner [M N K]\n";
    return 1;
  }

  try
  {
    return runBenchmark(M, N, K);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Runtime error during '" << g_stage << "': " << e.what() << "\n";
    std::cout << "{\"status\":\"runtime_error\",\"stage\":" << jstr(g_stage)
              << ",\"message\":" << jstr(e.what()) << "}\n";
    return 3;
  }
}
