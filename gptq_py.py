import torch
from torch import nn

# from exllamav2.ext import exllamav2_ext as ext_c

from quant import Quantizer, collate_quantizers

from gptq import accumulate_hessian


class HessianHook:
    def __init__(self):
        super().__init__()
        self.hessian: torch.Tensor | None = None
        self.n_samples: int = 0
        self.hessian_inv: torch.Tensor | None = None
        self.perm: torch.Tensor | None = None
        self.perm_inv: torch.Tensor | None = None

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, use_kernel: bool = True) -> None:
        """
        inp: (..., N, D)
        """
        assert self.hessian_inv is None
        assert not hasattr(inp, 'fake_mode')
        if inp.dim() <= 2:
            inp = inp[None]
        self.n_samples += len(inp)
        if self.hessian is None:
            self.hessian = torch.zeros(inp.size(-1), inp.size(-1), dtype=torch.float32, device=inp.device)
        inp = inp.flatten(end_dim=-2)
        if not use_kernel:
            inp = inp.to(dtype=torch.float32)
            torch.addmm(self.hessian, inp.t(), inp, beta=1, alpha=1, out=self.hessian)  # self.hessian += inp.t() @ inp
        accumulate_hessian(self.hessian, inp)

    @torch.no_grad()
    def invert(self, damp_ratio: float = 1e-2, act_order: bool = True) -> torch.Tensor:
        self.hessian = self.hessian * (2. / self.n_samples)

        dead: torch.Tensor = self.hessian.diag() == 0.
        self.hessian[dead, dead] = 1.

        if act_order:
            self.perm = self.hessian.diag().argsort(descending=True)
            self.hessian = self.hessian[self.perm][:, self.perm]
            self.perm_inv = self.perm.argsort()

        diag_indicies = torch.arange(len(self.hessian), device=self.hessian.device)
        damp = damp_ratio * self.hessian.diag().mean()

        max_try = 100
        while (max_try := max_try - 1) >= 0:
            self.hessian[diag_indicies, diag_indicies] += damp
            self.hessian_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(self.hessian)),
                                                     upper=True)
            if not self.hessian_inv.isnan().any():
                break
        assert max_try >= 0
        return self.hessian_inv


class HessianHookWrapper(nn.Module):
    def __init__(self, hessian_hook: HessianHook):
        super().__init__()
        self.hessian_hook: HessianHook = hessian_hook

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        self.hessian_hook.add_batch(hidden_states)
        raise ValueError


@torch.no_grad()
def gptq_quant(
        weight: torch.Tensor,
        hessian_hook: HessianHook | None,
        group_sizes: torch.Tensor,
        group_bit_widths: torch.Tensor,
        scale_bit_width: torch.Tensor = None,
        gptq_use_kernel: bool = True,
        gptq_block_sizes: torch.Tensor = None,
        quant_symmetric: bool = False,
        quant_mse: bool = True,
        quant_max_shrink: float = .8,
        quant_n_grid: int = 100,
        quant_norm: float = 2.4,
        quant_use_kernel: bool = False,
) -> dict:
    weight_ref = weight
    weight = weight.to(dtype=torch.float32).clone()
    n_rows, n_columns = weight.shape
    device = weight.device

    if hessian_hook is not None:
        dead: torch.Tensor = hessian_hook.hessian.diag() == 0.
        weight[:, dead] = 0.
        if hessian_hook.perm is not None:
            weight = weight[:, hessian_hook.perm]

    group_ids: list[int] = [0] + group_sizes.cumsum(dim=-1).tolist()
    quantizers = []

    if hessian_hook is None:
        # disable gptq and round to the nearest
        quant = torch.empty_like(weight)
        qweight = torch.empty_like(weight, dtype=torch.int16)
        for bit_width, i1, i2 in zip(group_bit_widths, group_ids[:-1], group_ids[1:]):
            quantizer = Quantizer()
            quantizer.find_params(weight[:, i1:i2], bit_width=bit_width, sym=quant_symmetric,
                                  scale_bit_width=scale_bit_width)
            if quant_mse:
                quantizer.mse(weight[:, i1:i2], max_shrink=quant_max_shrink, n_grid=quant_n_grid, norm=quant_norm,
                              use_kernel=quant_use_kernel)
            quantizers.append(quantizer)
            qweight[:, i1:i2] = qw = quantizer.quantize(weight[:, i1:i2])
            quant[:, i1:i2] = quantizer.dequantize(qw)
        error = torch.tensor(0., device=device)
    elif gptq_use_kernel:
        # assume gptq_block_sizes is the same as quantization group_sizes
        torch.cuda.set_device(device)  # exllama kernel requirement
        weight_t = weight.t().contiguous()
        quant_t = torch.empty_like(weight_t)
        error_t = torch.empty_like(weight_t)
        qweight_t = torch.empty_like(weight_t, dtype=torch.int16)
        for bit_width, i1, i2 in zip(group_bit_widths, group_ids[:-1], group_ids[1:]):
            quantizer = Quantizer()
            quantizer.find_params(weight[:, i1:i2], bit_width=bit_width, sym=quant_symmetric,
                                  scale_bit_width=scale_bit_width)
            if quant_mse:
                quantizer.mse(weight[:, i1:i2], max_shrink=quant_max_shrink, n_grid=quant_n_grid, norm=quant_norm,
                              use_kernel=quant_use_kernel)
            quantizers.append(quantizer)
            scale = quantizer.scale[:, 0].contiguous()
            qzero = quantizer.qzero[:, 0].contiguous()
            maxq = float(quantizer.maxq)
            hessian_inv = hessian_hook.hessian_inv.contiguous()
            ext_c.quantize_range(quant_t, scale, qweight_t, qzero, maxq, hessian_inv, weight_t, error_t, i1, i2)
            # quant: (C, R), float32, out
            # scale: (R), float32, in
            # qweight: (C, R), int16, out
            # qzero: (R), float32, in
            # maxq: float32, in
            # hessian_inv: (C, C), float32, int
            # weights: (C, R), float32, contiguous, in/out
            # error: (C, R), float32, out
            # a: int, in
            # b: int, in
        quant = quant_t.t()
        qweight = qweight_t.t()
        error = error_t.t()
    else:
        quant = torch.empty_like(weight)
        qweight = torch.empty_like(weight, dtype=torch.int16)
        error = torch.empty_like(weight)
        block_ids: list[int] = [0] + gptq_block_sizes.cumsum(dim=-1).tolist()
        cur_group = 0
        quantizer = None
        for i1, i2 in zip(block_ids[:-1], block_ids[1:]):
            weight_block = weight[:, i1:i2].clone()
            error_block = torch.empty_like(weight_block)
            for j in range(i1, i2):
                if j == group_ids[cur_group]:
                    quantizer = Quantizer()
                    quantizer.find_params(
                        weight[:, j: j + group_sizes[cur_group]],
                        bit_width=group_bit_widths[cur_group],
                        sym=quant_symmetric,
                        scale_bit_width=scale_bit_width,
                    )
                    if quant_mse:
                        quantizer.mse(weight[:, i1:i2], max_shrink=quant_max_shrink, n_grid=quant_n_grid,
                                      norm=quant_norm, use_kernel=quant_use_kernel)
                    quantizers.append(quantizer)
                    cur_group += 1
                w = weight_block[:, j - i1: j - i1 + 1]
                qweight[:, j: j + 1] = qw = quantizer.quantize(w)
                quant[:, j: j + 1] = q = quantizer.dequantize(qw)
                error[:, j: j + 1] = error_block[:, j - i1: j - i1 + 1] \
                    = err = (w - q) / hessian_hook.hessian_inv[j, j]
                # here is different from paper and kernel: no need to update weight_block[:, j - i1]
                weight_block[:, j - i1 + 1:] -= err @ hessian_hook.hessian_inv[j: j + 1, j + 1: i2]
            weight[:, i2:] -= error_block @ hessian_hook.hessian_inv[i1:i2, i2:]

    metrics = {
        'gptq_error': ((error ** 2.).mean()).item(),
        'gptq_norm': (((
                           (weight_ref / hessian_hook.hessian_inv.diag()) if hessian_hook is not None else error
                        ) ** 2.).mean()).item(),
    }

    indicies_c = torch.arange(0, n_columns, dtype=torch.int32, device=device)
    groups = [
        (bw, gs, -ig, ig, indicies_c[i1:i2],)
        for bw, gs, ig, i1, i2 in zip(
            group_bit_widths, group_sizes, range(len(group_sizes)), group_ids[:-1], group_ids[1:]
        )
    ]
    if scale_bit_width is not None:
        groups.sort(key=lambda g: g[:3], reverse=True)
    indicies_c = torch.cat([g[-1] for g in groups])
    indicies_g = torch.tensor([g[-2] for g in groups], dtype=torch.int32, device=device)

    quant_meta_g = collate_quantizers(*quantizers)

    quant = quant[:, indicies_c]
    qweight = qweight.to(dtype=torch.uint8)[:, indicies_c].cpu()  # (R, C)
    scale = quant_meta_g['scale'].to(dtype=torch.float16)[:, indicies_g].cpu()  # (R, G)
    qzero = quant_meta_g['qzero'].to(dtype=torch.uint8)[:, indicies_g].cpu()  # (R, G)
    qscale = quant_meta_g['qscale'].to(dtype=torch.uint8)[:, indicies_g].cpu() if 'qscale' in quant_meta_g else None  # (R, G)
    sscale = quant_meta_g['sscale'].to(dtype=torch.float16)[0, indicies_g].cpu() if 'sscale' in quant_meta_g else None  # (G)
    if hessian_hook is not None and hessian_hook.perm is not None:
        perm = hessian_hook.perm.to(dtype=torch.int16)[indicies_c].cpu()  # (C)
    else:
        perm = torch.arange(n_columns, dtype=torch.int16, device=device)[indicies_c].cpu()
    perm_inv = perm.argsort().to(dtype=torch.int16).cpu()  # (C)
    group_sizes = group_sizes.to(dtype=torch.int16)[indicies_g].cpu()  # (G)
    group_bit_widths = group_bit_widths.to(dtype=torch.uint8)[indicies_g].cpu()  # (G)
    scale_bit_width = scale_bit_width.to(dtype=torch.uint8).cpu() if scale_bit_width is not None else None  # ()

    if perm_inv is not None:
        quant = quant[:, perm_inv.to(dtype=torch.int32)]
    metric_norm = 2
    diff = ((quant.to(dtype=torch.float32) - weight_ref.to(dtype=torch.float32)).abs() ** metric_norm).mean()
    metrics.update({
        f'l{metric_norm}_error': diff.item(),
        f'l{metric_norm}_norm': (weight_ref.abs() ** metric_norm).mean().item(),
    })

    quant_meta = {
        'qweight': qweight,
        'scale': scale,
        'qzero': qzero,
        'qscale': qscale,
        'sscale': sscale,
        'perm_inv': perm_inv,
        'group_sizes': group_sizes,
        'group_bit_widths': group_bit_widths,
        'scale_bit_width': scale_bit_width,
    }
    return {'quant_meta': quant_meta, 'metrics': metrics}
