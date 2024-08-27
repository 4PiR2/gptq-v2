import torch  # ImportError: libc10.so: cannot open shared object file: No such file or directory

import gptq_c


def accumulate_hessian(mat_hessian, mat_input):
    gptq_c.accumulate_hessian(mat_hessian, mat_input)
