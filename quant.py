import torch
from torch import nn

# from exllamav2.ext import exllamav2_ext as ext_c


EPSILON = 1e-12


def quantize(x: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    """
    quantize each row using a scale and a zero level
    x: (..., C, ...), weight
    scale: (..., 1, ...)
    qzero: (..., 1, ...)
    maxq: ()
    return: (..., C, ...), quantized x, same dtype
    """
    return ((x / scale).round() + qzero).clamp(0., maxq)


def dequantize(qx: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor) -> torch.Tensor:
    """
    dequantize each row using a scale and a zero level
    qx: (..., C, ...)
    scale: (..., 1, ...)
    qzero: (..., 1, ...)
    return: (..., C, ...)
    """
    return (qx - qzero) * scale


def dequantize_quantized(x: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    """
    truncate the numbers by first quantize then dequantize
    """
    return dequantize(quantize(x, scale, qzero, maxq), scale, qzero)


def quantize2(x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    """
    quadratic quantization, quantize each row using a scale
    Quantize a column of scales for EXL2 format. Should only be used in addition to quantize().
    x: (..., C, ...), > 0.
    scale: (..., 1, ...)
    maxq: ()
    return: (..., C, ...), quantized x, same dtype
    !!! Note: When calling ext_c.pack_rows_4(), the kernel function will -1 on all qscale elements.
    The internal values are always reduced by 1 in kernel codes
    """
    return (x / scale).sqrt().round().clamp(1., maxq + 1.)


def dequantize2(qx: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    quadratic dequantization, quantize each row using a scale
    Dequantize a column of scales for EXL2 format. Should only be used in addition to dequantize().
    qx: (..., C, ...)
    scale: (..., 1, ...)
    return: (..., C, ...)
    """
    return qx * qx * scale


def dequantize2_quantized2(x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    """
    truncate the numbers by first quantize then dequantize, using quadratic quantization
    """
    return dequantize2(quantize2(x, scale, maxq), scale)


class Quantizer:
    def __init__(self, scale: torch.Tensor = None, qzero: torch.Tensor = None, maxq: torch.Tensor = None,
                 qscale: torch.Tensor = None, sscale: torch.Tensor = None, smaxq: torch.Tensor = None):
        self.scale: torch.Tensor = scale  # (..., R, 1)
        self.qzero: torch.Tensor = qzero  # (..., R, 1)
        self.maxq: torch.Tensor = maxq  # ()

        # only for EXL2 format:
        self.qscale: torch.Tensor = qscale  # (..., R, 1)
        self.sscale: torch.Tensor = sscale  # (..., 1, 1)
        self.smaxq: torch.Tensor = smaxq  # ()

        if self.scale is None and self.qscale is not None and self.sscale is not None:
            self.scale = dequantize2(self.qscale, self.sscale)
        elif self.qscale is None and self.scale is not None and self.sscale is not None and self.smaxq is not None:
            self.qscale = quantize2(self.scale, self.sscale, self.smaxq)

    def find_params(
            self, x: torch.Tensor, bit_width: torch.Tensor, sym: bool = False, scale_bit_width: torch.Tensor = None,
    ) -> None:
        """
        find quantization metadata over dim=-1
        x: (..., R, C), weight
        bit_width: int
        sym: bool, whether to set qzero to the middle
        scale_bit_width: (), int, used in EXL2, only works with sym = True
        """
        if scale_bit_width is not None and scale_bit_width > 0:
            sym = True

        self.maxq = 2. ** bit_width - 1.  # ()

        if sym:
            self.scale = x.abs().max(dim=-1, keepdim=True)[0] * (2. / self.maxq) + EPSILON  # (..., R, 1)
            self.qzero = torch.full_like(self.scale, ((self.maxq + 1.) * .5).round())  # (..., R, 1)

            if scale_bit_width is not None and scale_bit_width > 0:
                self.smaxq = 2. ** scale_bit_width - 1.  # ()
                self.sscale = self.scale.max(dim=-2, keepdim=True)[0] / (self.smaxq + 1.) ** 2. + EPSILON  # (..., 1, 1)
                self.qscale = quantize2(self.scale, self.sscale, self.smaxq)  # (..., R, 1)
                self.scale = dequantize2(self.qscale, self.sscale)  # (..., R, 1)

        else:
            x_max = x.max(dim=-1, keepdim=True)[0].relu()  # (..., R, 1)
            x_min = -(-x.min(dim=-1, keepdim=True)[0]).relu()  # (..., R, 1)
            self.scale = (x_max - x_min) / self.maxq + EPSILON  # (..., R, 1)
            self.qzero = (-x_min / self.scale).round()  # (..., R, 1)

    def mse(self, x: torch.Tensor, max_shrink: float = .8, n_grid: int = 100, norm: float = 2.4,
            use_kernel: bool = False) -> None:
        """
        refine quantization metadata
        mean squared error?
        x: (..., R, C), weight
        use_kernel: bool, use exllamav2 kernel, only works with 2D input tensor, same shrinking ratio for all rows
        """
        device: torch.device = x.device
        if use_kernel:
            assert x.dim() == 2
            torch.cuda.set_device(device)  # exllama kernel requirement
            min_p, max_p = .75, 1.
            x = x.transpose(-2, -1).contiguous()  # (..., C, R)
            err = torch.zeros(n_grid + 1, 128, dtype=torch.float32, device=device)
            scale = self.scale[..., 0].contiguous()  # (..., R)
            qzero = self.qzero[..., 0].contiguous()  # (..., R)
            maxq = float(self.maxq)
            ext_c.quantize_err(x, err, scale, qzero, maxq, norm, min_p, max_p, n_grid)
            # x: (C, R), float32, contiguous, in
            # err: (n_grid + 1, 128), float32, out
            # scale: (R), float32, in
            # qzero: (R), float32, in
            # maxq, norm, min_p, max_p: float32, in
            # n_grid: int, in
            best_pi = err.sum(dim=-1).argmin() / n_grid  # ()
            best_p = max_p * best_pi + min_p * (1. - best_pi)  # ()
            self.scale *= best_p  # (..., R, 1)
            if self.sscale is not None:
                self.sscale *= best_p  # (..., 1, 1)
        else:
            p = 1. - torch.arange(0., max_shrink, 1. / n_grid, device=device)  # (Q)
            q = dequantize_quantized(x[..., None], self.scale[..., None] * p, self.qzero[..., None], self.maxq)
            # (..., R, C, Q)
            err_argmin = (q - x[..., None]).abs().pow(norm).sum(dim=-2).argmin(dim=-1, keepdim=True)  # (..., R, 1)
            self.scale *= p[err_argmin]  # (..., R, 1)
            if self.sscale is not None:
                self.sscale = self.scale.max(dim=-2, keepdim=True)[0] / (self.smaxq + 1.) ** 2. + EPSILON
                # (..., 1, 1)
                q = dequantize2_quantized2(self.scale[..., None], self.sscale[..., None] * p, self.smaxq)
                # (..., R, 1, Q)
                err_argmin = (q - self.scale[..., None]).abs().pow(norm).sum(dim=-3).argmin(dim=-1, keepdim=True)
                # (..., 1, 1)
                self.sscale *= p[err_argmin]  # (..., 1, 1)
                self.qscale = quantize2(self.scale, self.sscale, self.smaxq)  # (..., R, 1)
                self.scale = dequantize2(self.qscale, self.sscale)  # (..., R, 1)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., R, C), weight
        """
        return quantize(x, self.scale, self.qzero, self.maxq)

    def dequantize(self, qx: torch.Tensor) -> torch.Tensor:
        """
        qx: (..., R, C), qweight
        """
        return dequantize(qx, self.scale, self.qzero)

    def dequantize_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., R, C), weight
        """
        return dequantize_quantized(x, self.scale, self.qzero, self.maxq)


def collate_quantizers(*args):
    """
    aggregate quantizer information
    input: Quantizer
    """
    # turn object into dicts
    tensor_dicts = []
    for a in args:
        tensor_dict = {}
        for attr_name in dir(a):
            if attr_name.startswith('_'):
                continue
            attr_value = getattr(a, attr_name)
            if isinstance(attr_value, torch.Tensor):
                tensor_dict[attr_name] = attr_value
        tensor_dicts.append(tensor_dict)

    # aggregate by keys
    aggregate_dict = {}
    for d in tensor_dicts:
        for key, value in d.items():
            if key in aggregate_dict:
                aggregate_dict[key].append(value)
            else:
                aggregate_dict[key] = [value]

    # concatenate tensors
    result = {}
    for key, value in aggregate_dict.items():
        if value[0].dim():
            result[key] = torch.cat(value, dim=-1)
        else:
            result[key] = torch.stack(value)  # stack zero-dimension tensors
    return result


def construct_matrix(
        qweight: torch.Tensor,  # (R, C), int16
        scale: torch.Tensor,  # (R, G), float16
        qzero: torch.Tensor,  # (R, G) or (G) or (), int16
        group_sizes: torch.Tensor,  # (G), int16
) -> torch.Tensor:
    """
    reconstruct matrix from aggregated quantizer information
    """
    qzero = qzero.expand(scale.shape)
    group_ids: list[int] = [0] + group_sizes.cumsum(dim=-1).tolist()
    weight = torch.empty_like(qweight, dtype=torch.float16, device=qweight.device)
    for k in range(len(group_ids) - 1):
        i1, i2 = group_ids[k], group_ids[k + 1]
        weight[:, i1:i2] = Quantizer(scale=scale[:, k:k+1], qzero=qzero[:, k:k+1]).dequantize(qweight[:, i1:i2])
    return weight


def construct_matrix_2(
        qweight: torch.Tensor,  # (R, C), int16
        qzero: torch.Tensor,  # (R, G) or (G) or (), int16
        qscale: torch.Tensor,  # (R, G), int16
        sscale: torch.Tensor,  # (G), float16
        group_sizes: torch.Tensor,  # (G), int16
) -> torch.Tensor:
    """
    reconstruct matrix from aggregated quantizer information (EXL2)
    """
    qzero = qzero.expand(qscale.shape)
    sscale = sscale.expand(1, sscale.size(-1))
    group_ids: list[int] = [0] + group_sizes.cumsum(dim=-1).tolist()
    weight = torch.empty_like(qweight, dtype=torch.float16, device=qweight.device)
    for k in range(len(group_ids) - 1):
        i1, i2 = group_ids[k], group_ids[k + 1]
        weight[:, i1:i2] = Quantizer(
            qzero=qzero[:, k:k+1], qscale=qscale[:, k:k+1], sscale=sscale[:, k:k + 1],
        ).dequantize(qweight[:, i1:i2])
    return weight


def reconstruct_nn_linear(quant_meta: dict, device: torch.device = torch.device('cpu')) -> nn.Linear:
    qweight = quant_meta['qweight'].to(dtype=torch.int16, device=device)
    qzero = quant_meta['qzero'].to(dtype=torch.int16, device=device)
    group_sizes = quant_meta['group_sizes'].to(dtype=torch.int16, device=device)
    if 'sscale' in quant_meta and quant_meta['sscale'] is not None:
        qscale = quant_meta['qscale'].to(dtype=torch.int16, device=device)
        sscale = quant_meta['sscale'].to(dtype=torch.float16, device=device)
        weight = construct_matrix_2(qweight, qzero, qscale, sscale, group_sizes)
    else:
        scale = quant_meta['scale'].to(dtype=torch.float16, device=device)
        weight = construct_matrix(qweight, scale, qzero, group_sizes)
    if 'perm_inv' in quant_meta and quant_meta['perm_inv'] is not None:
        perm_inv = quant_meta['perm_inv'].to(dtype=torch.int32, device=device)
        weight = weight[:, perm_inv]
    nn_linear = nn.Linear(*weight.shape[::-1], bias=False, dtype=torch.float16, device=weight.device)
    nn_linear.weight.data = weight
    nn_linear.eval()
    return nn_linear
