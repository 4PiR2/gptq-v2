import os

import setuptools
from torch.utils import cpp_extension


CUTLASS_PATH = '/nfs/scistore19/alistgrp/jiachen/cutlass'

setuptools.setup(
    name='gptq',
    version='0.0.1',
    description='Fast GPTQ Package',
    install_requires=['torch'],
    packages=setuptools.find_packages(exclude=['docs', 'examples', 'tests']),
    ext_modules=[cpp_extension.CUDAExtension(
        'gptq_c',
        [
            'gptq/gemm.cpp',
            'gptq/gemm_kernel.cu',
        ],
    )],
    include_dirs=[
        os.path.join(CUTLASS_PATH, 'include'),
        os.path.join(CUTLASS_PATH, 'tools', 'util', 'include'),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)