import logging
import time

import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaForCausalLM


from data_utils import get_dataloader
from model_utils import Catcher, find_submodules_by_types, extract_dependencies, find_and_replace_submodule_by_name, \
    move_to_device, RecorderWrapper
from quant import reconstruct_nn_linear
from gptq_py import HessianHook, gptq_quant


def get_llama(model_path: str) -> LlamaForCausalLM:
    nn.init.kaiming_uniform_ = nn.init.uniform_ = nn.init.normal_ = lambda *args, **kwargs: None
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype='auto')
    return model


@torch.no_grad()
def get_initial_inputs(
        model: LlamaForCausalLM, encodings: torch.Tensor, device: torch.device, batch_size: int = 1,
) -> tuple[torch.Tensor, dict]:
    """
    catch first layer's input
    encodings: (B, N=SeqLen), int64
    inps: inputs, (B, N=SeqLen, D), float16
    attention_mask = None
    position_ids: (1, N=SeqLen), int64
    past_key_value: None
    output_attentions: bool = False
    use_cache: bool = False
    cache_position: (N=SeqLen), int64
    position_embeddings: tuple, FP16, (1, N=SeqLen, 128), (1, N=SeqLen, 128)
    """
    catcher: Catcher = Catcher()
    gpt_blocks: nn.ModuleList = model.model.layers

    gpt_block_0, gpt_blocks[0] = gpt_blocks[0], catcher
    use_cache, model.config.use_cache = model.config.use_cache, False
    model.model.embed_tokens.to(device=device)
    model.model.rotary_emb.to(device=device)

    for bi in range(0, len(encodings), batch_size):
        try:
            model(encodings[bi : bi + batch_size].to(device=device))
        except ValueError:
            pass

    inps: torch.Tensor = torch.cat(catcher.inps, dim=0)
    inp_kwargs: dict = catcher.inp_kwargs

    gpt_blocks[0] = gpt_block_0
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cpu()
    model.config.use_cache = use_cache
    return inps.cpu(), move_to_device(inp_kwargs, torch.device('cpu'))


@torch.no_grad()
def quantize_llama(model, encodings, device, batch_size=8):
    """
    start quantize
    """
    _input, _output = 'input', 'output'
    cpu_device: torch.device = torch.device('cpu')
    inps, inp_kwargs = get_initial_inputs(model, encodings, device, batch_size)
    use_cache, model.config.use_cache = model.config.use_cache, False

    results: dict[str, dict] = {
        'data': {},  # dict[str, list]
        'metrics': {},  # dict[str, list[dict[str, Any]]]
    }

    gpt_blocks: nn.ModuleList = model.model.layers

    for gi, gpt_block in enumerate(gpt_blocks):
        start_time = time.time()

        dependency_info: list = extract_dependencies(gpt_block, [nn.Linear], inps.shape, inps.dtype, cpu_device, inp_kwargs)

        wrapper_layers_dict: dict[str, RecorderWrapper] = {}
        for linear_layer_name, linear_layer_module in find_submodules_by_types(gpt_block, [nn.Linear]).items():
            linear_layer_wrapper = RecorderWrapper(linear_layer_module)
            linear_layer_wrapper.stage = RecorderWrapper.stage_fake_mode
            wrapper_layers_dict[linear_layer_name] = linear_layer_wrapper
            find_and_replace_submodule_by_name(gpt_block, linear_layer_name, linear_layer_wrapper)

        gpt_block.input_layernorm.to(device=device)
        gpt_block.post_attention_layernorm.to(device=device)

        for di, (quantizing_layer_names, to_release_layer_names) in enumerate(dependency_info):
            if quantizing_layer_names == [_output]:
                break

            hessian_hook = HessianHook()
            for quantizing_layer_name in quantizing_layer_names:
                quantizing_layer_wrapper = wrapper_layers_dict[quantizing_layer_name]
                quantizing_layer_wrapper.hessian_hook = hessian_hook
                quantizing_layer_wrapper.stage = RecorderWrapper.stage_hessian_accumulation
            inp_kwargs_gpu = move_to_device(inp_kwargs, device=device)
            for bi in range(0, len(inps), batch_size):
                try:
                    gpt_block(inps[bi : bi + batch_size].to(device=device), **inp_kwargs_gpu)
                except ValueError:
                    pass
            del inp_kwargs_gpu
            hessian_hook.invert(damp_ratio=1e-2, act_order=True)

            for quantizing_layer_name in quantizing_layer_names:
                assert quantizing_layer_name not in [_input, _output]
                quantizing_layer_wrapper = wrapper_layers_dict[quantizing_layer_name]
                weight = quantizing_layer_wrapper.module.weight.to(device=device)
                n_out_features, n_in_features = weight.shape
                group_sizes: torch.Tensor = torch.full([n_in_features // 128], 128, dtype=torch.int32, device=device)
                group_bit_widths: torch.Tensor = torch.full([n_in_features // 128], 4, dtype=torch.int32, device=device)
                gptq_result: dict[str, dict] = gptq_quant(
                    weight=weight,
                    hessian_hook=hessian_hook,
                    group_sizes=group_sizes,
                    group_bit_widths=group_bit_widths,
                    scale_bit_width=None,
                    gptq_use_kernel=False,
                    gptq_block_sizes=group_sizes,
                    quant_symmetric=False,
                    quant_mse=True,
                    quant_max_shrink=.8,
                    quant_n_grid=100,
                    quant_norm=2.4,
                    quant_use_kernel=False,
                )
                quant_meta: dict[str, torch.Tensor | None] = gptq_result['quant_meta']
                quantizing_layer_wrapper.hessian_hook = None
                quantizing_layer_wrapper.module = reconstruct_nn_linear(quant_meta, device).cpu()
                metrics: dict[str, float] = gptq_result['metrics']
                print(gi, quantizing_layer_name, metrics)

            for quantizing_layer_name in quantizing_layer_names:
                quantizing_layer_wrapper = wrapper_layers_dict[quantizing_layer_name]
                quantizing_layer_wrapper.module.to(device=device)
                quantizing_layer_wrapper.stage = RecorderWrapper.stage_recording_mode
            inp_kwargs_gpu = move_to_device(inp_kwargs, device=device)
            for bi in range(0, len(inps), batch_size):
                gpt_block(inps[bi : bi + batch_size].to(device=device), **inp_kwargs_gpu)
            del inp_kwargs_gpu
            for quantizing_layer_name in quantizing_layer_names:
                quantizing_layer_wrapper = wrapper_layers_dict[quantizing_layer_name]
                quantizing_layer_wrapper.module.cpu()
                quantizing_layer_wrapper.stage = RecorderWrapper.stage_replay_mode

            for to_release_layer_name in to_release_layer_names:
                if to_release_layer_name == _input:
                    inps = RecorderWrapper.fake_mode.from_tensor(inps)
                    continue
                to_release_layer_wrapper = wrapper_layers_dict[to_release_layer_name]
                to_release_layer_wrapper.outs = []
                to_release_layer_wrapper.stage = RecorderWrapper.stage_fake_mode

        outs = torch.empty(*inps.shape, dtype=inps.dtype, device=cpu_device)
        inp_kwargs_gpu = move_to_device(inp_kwargs, device=device)
        for bi in range(0, len(inps), batch_size):
            (outs[bi : bi + batch_size],) = gpt_block(inps[bi : bi + batch_size].to(device=device), **inp_kwargs_gpu)
        del inp_kwargs_gpu
        inps = outs

        gpt_block.input_layernorm.cpu()
        gpt_block.post_attention_layernorm.cpu()

        for linear_layer_name, linear_layer_wrapper in wrapper_layers_dict.items():
            find_and_replace_submodule_by_name(gpt_block, linear_layer_name, linear_layer_wrapper.module)

    return None


@torch.no_grad()
def evaluate_llama(
        model: LlamaForCausalLM, encodings: torch.Tensor, device: torch.device, batch_size: int = 8,
) -> torch.Tensor:
    inps, inp_kwargs = get_initial_inputs(model, encodings, device, batch_size)
    inp_kwargs = move_to_device(inp_kwargs, device=device)
    use_cache, model.config.use_cache = model.config.use_cache, False

    layers: nn.ModuleList = model.model.layers
    outs = torch.empty_like(inps)

    for i, layer in tqdm(enumerate(layers)):
        layer.to(device)
        for j in range(0, len(inps), batch_size):
            (outs[j:j+batch_size],) = layer(inps[j:j+batch_size].to(device=device), **inp_kwargs)
        layer.cpu()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm.to(device=device)
        for j in range(0, len(inps), batch_size):
            outs[j:j+batch_size] = model.model.norm(inps[j:j+batch_size].to(device=device))
        inps, outs = outs, inps

    model.lm_head = model.lm_head.to(device=device)
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for i in range(0, len(inps), batch_size):
        lm_logits = model.lm_head(inps[i:i+batch_size].to(device=device))  # (B, N=SeqLen, D)
        shift_logits = lm_logits[:, :-1, :]  # (B, N=SeqLen-1, D)
        shift_labels = encodings[i:i+batch_size, 1:].to(device=device)  # (B, N=SeqLen-1)
        neg_log_likelihood = loss_fct(shift_logits.flatten(end_dim=-2), shift_labels.flatten())
        nlls.extend([neg_log_likelihood] * len(lm_logits))
    ppl = torch.stack(nlls).to(dtype=torch.float32).mean().exp()

    model.model.norm.cpu()
    model.config.use_cache = use_cache
    return ppl


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    model = get_llama(args.model)
    model.eval()

    dataloader = get_dataloader(
       name=args.dataset, split='train', seqlen=2048, n_samples=args.nsamples, model_path=args.model, seed=args.seed, cache_dir='./cache/datasets'
    )

    DEV = torch.device('cuda:0')

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = quantize_llama(model, dataloader, DEV)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        testloader = get_dataloader(
            name=dataset, split='test', seqlen=2048, n_samples=args.nsamples, model_path=args.model, seed=args.seed, cache_dir='./cache/datasets'
        )
        print(dataset)
        ppl = evaluate_llama(model, testloader, DEV)
        print(ppl.item())
