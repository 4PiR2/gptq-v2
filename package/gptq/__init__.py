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
