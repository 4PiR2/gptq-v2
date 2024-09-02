// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md


#include "cutlass/gemm/device/gemm.h"


template<typename InputDtype, typename OutputDtype>
inline int accumulate_hessian_cuda(
    void* mat_hessian,
    const void* mat_input,
    const int size_hidden,
    const int size_batch
) {
    using ElementAccumulator = OutputDtype;

    using Gemm = cutlass::gemm::device::Gemm<
        InputDtype,
        cutlass::layout::ColumnMajor,
        InputDtype,
        cutlass::layout::RowMajor,
        OutputDtype,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,  // 128
        cutlass::gemm::GemmShape<64, 64, 32>,  // 64
        cutlass::gemm::GemmShape<16, 8, 16>,  // size of tensor core; do not change
        cutlass::epilogue::thread::LinearCombination<
            OutputDtype,
            128 / cutlass::sizeof_bits<OutputDtype>::value,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3  // number of stages, usually 2-4
    >;

    // Create GEMM operation
    Gemm gemm_op;

    // Define GEMM arguments
    typename Gemm::Arguments args(
        {size_hidden, size_hidden, size_batch},  // problem size
        {(InputDtype*) mat_input, size_hidden},  // tensor A (with leading dimension)
        {(InputDtype*) mat_input, size_hidden},  // tensor B (with leading dimension)
        {(OutputDtype*) mat_hessian, size_hidden},  // tensor C (output with leading dimension)
        {(OutputDtype*) mat_hessian, size_hidden},  // tensor D (same as C)
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


int accumulate_hessian_fp16_fp32_cuda(
    void* mat_hessian,
    const void* mat_input,
    const int size_hidden,
    const int size_batch
) {
    return accumulate_hessian_cuda<cutlass::half_t, float>(mat_hessian, mat_input, size_hidden, size_batch);
}


int accumulate_hessian_bf16_fp32_cuda(
    void* mat_hessian,
    const void* mat_input,
    const int size_hidden,
    const int size_batch
) {
    return accumulate_hessian_cuda<cutlass::bfloat16_t, float>(mat_hessian, mat_input, size_hidden, size_batch);
}
