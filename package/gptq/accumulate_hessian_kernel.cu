#ifndef ACCUMULATE_HESSIAN_KERNEL_CUH
#define ACCUMULATE_HESSIAN_KERNEL_CUH


// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md


#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"


int accumulate_hessian_kernel(
    void* mat_hessian,
    const void* mat_input,
    const int size_hidden,
    const int size_batch
) {
    // Define the GEMM type
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor
    >;

    // Create GEMM operation
    Gemm gemm_op;

    // Define GEMM arguments
    typename Gemm::Arguments args(
        {size_hidden, size_hidden, size_batch},  // problem size
        {(cutlass::half_t*) mat_input, size_hidden},  // tensor A (with leading dimension)
        {(cutlass::half_t*) mat_input, size_hidden},  // tensor B (with leading dimension)
        {(float*) mat_hessian, size_hidden},  // tensor C (output with leading dimension)
        {(float*) mat_hessian, size_hidden},  // tensor D (same as C)
        {1.f, 1.f}  // scalars for alpha and beta
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
