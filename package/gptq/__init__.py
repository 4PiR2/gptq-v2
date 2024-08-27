import torch  # ImportError: libc10.so: cannot open shared object file: No such file or directory

import gptq_c


def mul(A, B, C):
    gptq_c.mul(A, B, C)
