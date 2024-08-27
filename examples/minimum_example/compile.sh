nvcc -I/nfs/scistore19/alistgrp/jiachen/cutlass/include -I/nfs/scistore19/alistgrp/jiachen/cutlass/tools/util/include cutlass_gemm.cu -o cutlass_gemm -lcuda -lcudart -arch=sm_86
