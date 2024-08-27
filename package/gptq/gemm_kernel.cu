#ifndef GEMM_KERNEL_CUH
#define GEMM_KERNEL_CUH


// https://github.com/NVIDIA/cutlass/blob/main/examples/05_batched_gemm/batched_gemm.cu
// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md


#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm_batched.h"


int gemm_kernel(
    const void* mat_A,
    const void* mat_B,
    void* mat_C,
    const int size_m,  // number of rows of A and C matrices
    const int size_n,  // number of columns of B and C matrices
    const int size_k,  // number of columns of A and rows of B matrices
    const int size_b  // number of batches
) {
    using ElementTypeA = cutlass::half_t;  // Data type for matrix elements
    using ElementTypeB = cutlass::half_t;  // Data type for matrix elements
    using ElementTypeC = float;  // Data type for matrix elements
    using LayoutTypeA = cutlass::layout::RowMajor;  // Layout of matrices (row-major)
    using LayoutTypeB = cutlass::layout::RowMajor;  // Layout of matrices (row-major)
    using LayoutTypeC = cutlass::layout::RowMajor;  // Layout of matrices (row-major)

    // Define the GEMM type
    using Gemm = cutlass::gemm::device::GemmBatched<ElementTypeA, LayoutTypeA, ElementTypeB, LayoutTypeB, ElementTypeC, LayoutTypeC>;

    // Create GEMM operation
    Gemm gemm_op;

    // Define GEMM arguments
    typename Gemm::Arguments args(
        {size_m, size_n, size_k},  // Problem size
        {(ElementTypeA*) mat_A, size_k},  // Tensor A (with leading dimension)
        size_m * size_k,  // batch_stride_A
        {(ElementTypeB*) mat_B, size_n},  // Tensor B (with leading dimension)
        size_k * size_n,  // batch_stride_B
        {(ElementTypeC*) mat_C, size_n},  // Tensor C (output with leading dimension)
        size_m * size_n,  // batch_stride_C
        {(ElementTypeC*) mat_C, size_n},  // Tensor D (same as C)
        size_m * size_n,  // batch_stride_D
        {1.f, 1.f},  // scalars for alpha and beta
        size_b  // batch count
    );

    // Run the GEMM operation
    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


#endif
