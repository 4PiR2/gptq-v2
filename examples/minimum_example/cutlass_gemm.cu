// Include necessary headers
#include <iostream>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

// Error checking macro for CUDA API calls
#define CUDA_CHECK(status)                                                   \
  if (status != cudaSuccess) {                                               \
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;   \
    exit(EXIT_FAILURE);                                                      \
  }

// Main function
int main() {
    using ElementType = float;  // Data type for matrix elements
    using LayoutType = cutlass::layout::RowMajor;  // Layout of matrices (row-major)

    const int M = 128;  // Number of rows of A and C matrices
    const int N = 128;  // Number of columns of B and C matrices
    const int K = 128;  // Number of columns of A and rows of B matrices

    // Define the GEMM type
    using Gemm = cutlass::gemm::device::Gemm<ElementType, LayoutType, ElementType, LayoutType, ElementType, LayoutType>;

    // Allocate and initialize host matrices
    std::vector<ElementType> A(M * K, 1.0f);  // Matrix A
    std::vector<ElementType> B(K * N, 1.0f);  // Matrix B
    std::vector<ElementType> C(M * N, 0.0f);  // Matrix C (output)

    // Allocate device memory
    ElementType* d_A;
    ElementType* d_B;
    ElementType* d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(ElementType)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(ElementType)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(ElementType)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(ElementType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(ElementType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), M * N * sizeof(ElementType), cudaMemcpyHostToDevice));

    // Create GEMM operation
    Gemm gemm_op;

    // Define GEMM arguments
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename Gemm::Arguments args(
        problem_size,    // Problem size
        {d_A, K},        // Tensor A (with leading dimension)
        {d_B, N},        // Tensor B (with leading dimension)
        {d_C, N},        // Tensor C (output with leading dimension)
        {d_C, N},        // Tensor D (same as C)
        {1.0f, 0.0f}     // Scalars for alpha and beta
    );

    // Run the GEMM operation
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return EXIT_FAILURE;
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(ElementType), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < M * N; ++i) {
        if (C[i] != K) {
            std::cerr << "Result verification failed at element " << i << ": " << C[i] << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "GEMM operation completed successfully!" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
