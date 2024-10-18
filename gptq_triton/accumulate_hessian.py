import torch
import triton
from triton import language as tl


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')


def _get_cuda_autotune_config() -> list[triton.Config]:
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=2, num_stages=5, num_ctas=1, maxnreg=None),
        # # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4, num_ctas=1, maxnreg=None),
    ]


@triton.autotune(
    configs=_get_cuda_autotune_config(),
    key=['size_hidden', 'size_batch', 'save_lower_only', 'compute_lower_only'],
    restore_value=['mat_hessian_ptr'],
)
@triton.jit
def accumulate_hessian_triton_kernel(
        mat_hessian_ptr,
        mat_input_ptr,
        size_hidden: int,
        size_batch: int,
        save_lower_only,
        compute_lower_only,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
) -> None:
    a_ptr, b_ptr, c_ptr = mat_input_ptr, mat_input_ptr, mat_hessian_ptr
    M, N, K = size_hidden, size_hidden, size_batch
    stride_am, stride_ak = 1, size_hidden
    stride_bk, stride_bn = size_hidden, 1
    stride_cm, stride_cn = size_hidden, 1

    # Kernel for computing the matmul C = A x B. A has shape (M, K), B has shape (K, N) and C has shape (M, N)

    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    is_upper = (pid_m + 1) * BLOCK_SIZE_M <= pid_n * BLOCK_SIZE_N
    if compute_lower_only and is_upper:
        return
    is_lower = pid_m * BLOCK_SIZE_M >= (pid_n + 1) * BLOCK_SIZE_N
    is_diag = not (is_lower or is_upper)

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # TODO: (unknown reason) tl.load c here makes the kernel 2x slow
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of fp32 values for higher accuracy.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.)
        # We accumulate along the K dimension.
        c = tl.dot(a, b, c)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c += tl.load(c_ptrs, mask=c_mask)
    tl.store(c_ptrs, c, mask=c_mask)

    if compute_lower_only and not (save_lower_only or is_diag):
        # Warning: BLOCK_SIZE_M:BLOCK_SIZE_N must be 1:n or n:1 for this kind of copying
        ct_ptrs = c_ptr + (offs_cm[:, None] * stride_cn + offs_cn[None, :] * stride_cm)
        tl.store(ct_ptrs, c, mask=c_mask)


def accumulate_hessian(
        mat_hessian: torch.Tensor,
        mat_input: torch.Tensor,
        save_lower_only: bool,
        compute_lower_only: bool = True,
) -> None:
    assert compute_lower_only or not save_lower_only, "compute_lower_only must be True when save_lower_only is True"
    torch.cuda.set_device(mat_input.device)
    size_batch, size_hidden = mat_input.shape
    grid = lambda meta: (triton.cdiv(size_hidden, meta['BLOCK_SIZE_M']) * triton.cdiv(size_hidden, meta['BLOCK_SIZE_N']),)
    accumulate_hessian_triton_kernel[grid](
        mat_hessian, mat_input,
        size_hidden, size_batch,
        save_lower_only,
        compute_lower_only,
    )


def torch_baseline(mat_hessian: torch.Tensor, mat_input: torch.Tensor) -> None:
    mat_input = mat_input.to(dtype=torch.float32)
    mat_hessian += mat_input.t() @ mat_input


def bad_baseline(mat_hessian: torch.Tensor, mat_input: torch.Tensor) -> None:
    mat_hessian += mat_input.t() @ mat_input


def unit_test(dtype: torch.dtype):
    import gptq

    torch.manual_seed(0)

    size_batch, size_hidden = 16384, 4096

    mat_input = torch.randn(size_batch, size_hidden, device='cuda', dtype=dtype)

    torch_output = torch.randn(size_hidden, size_hidden, device='cuda', dtype=torch.float32)
    torch_output = torch_output + torch_output.t()
    bad_output = torch_output.clone()
    cutlass_output = torch_output.clone()
    triton_output = torch_output.clone()
    triton_output_2 = torch_output.clone()
    triton_output_3 = torch_output.clone()

    torch_baseline(torch_output, mat_input)
    bad_baseline(bad_output, mat_input)
    gptq.accumulate_hessian(cutlass_output, mat_input)
    accumulate_hessian(triton_output, mat_input, save_lower_only=True, compute_lower_only=True)
    print(accumulate_hessian_triton_kernel.best_config)

    torch_output.tril_()
    bad_output.tril_()
    cutlass_output.tril_()
    triton_output.tril_()

    accumulate_hessian(triton_output_2, mat_input, save_lower_only=False, compute_lower_only=True)
    assert (triton_output == triton_output_2.tril()).all() and (triton_output_2 == triton_output_2.t()).all()
    accumulate_hessian(triton_output_3, mat_input, save_lower_only=False, compute_lower_only=False)
    assert (triton_output == triton_output_3.tril()).all() and (triton_output_3 == triton_output_3.t()).all()

    diff_b = bad_output - torch_output
    diff_c = cutlass_output - torch_output
    diff_t = triton_output - torch_output
    print(
        diff_b.abs().mean().item(),
        diff_c.abs().mean().item(),
        diff_t.abs().mean().item(),
        (triton_output - cutlass_output).abs().mean().item(),
        sep='\t',
    )


def benchmark(dtype: torch.dtype):
    from matplotlib import pyplot as plt
    import gptq

    configs = [triton.testing.Benchmark(
        x_names=['N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(8, 15)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['pytorch', 'bad', 'cutlass', 'triton', 'triton_l'],  # Label name for the lines
        line_names=['PyTorch (FP32)', 'PyTorch (FP16)', 'CUTLASS', 'Triton', 'Triton (Lower)'],  # Line styles
        plot_name='matmul-performance',  # Name for the plot, used also as a file name for saving the plot.
        args={},
        xlabel='N',  # Label name for the y-axis
        ylabel='TFLOPS',  # Label name for the y-axis
        x_log=True,
        y_log=True,
        color=None,
        styles=[('#1f77b4', '-'), ('#ff7f0e', '-'), ('#2ca02c', '-'), ('#d62728', '-'), ('#d62728', '--')],
    )]
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    @triton.testing.perf_report(configs)
    def _benchmark(N, K, provider):
        a = torch.randn(K, N, device='cuda', dtype=dtype)
        c = torch.randn(N, N, device='cuda', dtype=torch.float32)
        quantiles = [.5, .2, .8]
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_baseline(c, a), quantiles=quantiles)
        if provider == 'bad':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: bad_baseline(c, a), quantiles=quantiles)
        if provider == 'cutlass':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: gptq.accumulate_hessian(c, a), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: accumulate_hessian(c, a, save_lower_only=False, compute_lower_only=False), quantiles=quantiles)
        if provider == 'triton_l':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: accumulate_hessian(c, a, save_lower_only=False, compute_lower_only=True), quantiles=quantiles)
        perf = lambda ms: 2. * N * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    result_dfs = _benchmark.run(show_plots=False, print_data=True, return_df=True)
    plt.grid()
    plt.show()
    return result_dfs


if __name__ == '__main__':
    unit_test(dtype=torch.float16)
    unit_test(dtype=torch.bfloat16)
    benchmark(dtype=torch.float16)
    benchmark(dtype=torch.bfloat16)
