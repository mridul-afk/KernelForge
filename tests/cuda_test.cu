#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_kernel()
{
  printf("Hello from CUDA! Block %d, Thread %d\n",
         blockIdx.x, threadIdx.x);
}

int main()
{
  int device_count = 0;

  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: "
              << cudaGetErrorString(err) << '\n';
    return 1;
  }

  std::cout << "CUDA devices found: "
            << device_count << '\n';

  if (device_count == 0)
  {
    std::cerr << "No CUDA GPU detected.\n";
    return 1;
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "GPU: " << prop.name << '\n';
  std::cout << "Compute capability: "
            << prop.major << "." << prop.minor << '\n';
  std::cout << "Global memory: "
            << prop.totalGlobalMem / (1024 * 1024)
            << " MB\n";

  hello_kernel<<<1, 4>>>();

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    std::cerr << "Kernel launch error: "
              << cudaGetErrorString(err) << '\n';
    return 1;
  }

  err = cudaDeviceSynchronize();

  if (err != cudaSuccess)
  {
    std::cerr << "Kernel execution error: "
              << cudaGetErrorString(err) << '\n';
    return 1;
  }

  std::cout << "CUDA kernel executed successfully.\n";

  return 0;
}
