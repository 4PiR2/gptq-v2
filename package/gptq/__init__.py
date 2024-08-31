import torch  # ImportError: libc10.so: cannot open shared object file: No such file or directory

import gptq_c


def accumulate_hessian(mat_hessian: torch.Tensor, mat_input: torch.Tensor) -> None:
    match mat_input.dtype:
        case torch.float16:
            gptq_c.accumulate_hessian_fp16_fp32(mat_hessian, mat_input)
        case torch.bfloat16:
            gptq_c.accumulate_hessian_bf16_fp32(mat_hessian, mat_input)
        case _:
            raise NotImplementedError


def gptq_quantize_range(
    quant: torch.Tensor,
    scale: torch.Tensor,
    out_q: torch.Tensor,
    qzero: torch.Tensor,
    maxq: float,
    hessian_inv: torch.Tensor,
    weights: torch.Tensor,
    error: torch.Tensor,
    a: int,
    b: int,
) -> None:
    gptq_c.gptq_quantize_range(quant, scale, out_q, qzero, maxq, hessian_inv, weights, error, a, b)
