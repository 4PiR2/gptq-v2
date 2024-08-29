import os
import random

from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_dataloader(
    name: str, split: str = 'train', seqlen: int = 2048, n_samples: int = 256, model_path: str = '', seed: int = 0,
    cache_dir: str = '',
) -> torch.Tensor:
    """
    generate data samples and cache them
    """
    if cache_dir:
        cache_path = os.path.join(cache_dir, f'{name}_{split}_{seqlen:04}_{n_samples:04}_{seed:04}.pth')
        if os.path.exists(cache_path):
            dataloader = torch.load(cache_path, weights_only=True)
            return dataloader
    else:
        cache_path = ''
    match name:
        case 'wikitext2':
            dataloader = get_wikitext2(split, seqlen, n_samples, model_path, seed)
        case 'ptb':
            dataloader = get_ptb(split, seqlen, n_samples, model_path, seed)
        case 'c4':
            dataloader = get_c4(split, seqlen, n_samples, model_path, seed, new=False)
        case 'c4-new':
            dataloader = get_c4(split, seqlen, n_samples, model_path, seed, new=True)
        case 'mmlu':
            dataloader = get_mmlu(split, seqlen, n_samples, model_path, seed)
        case _:
            raise NotImplementedError
    if cache_dir:
        torch.save(dataloader, cache_path)
    return dataloader


def get_wikitext2(split: str, seqlen: int, n_samples: int, model_path: str, seed: int) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=False)
    dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split=split)
    input_ids = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt').input_ids  # (N=1, SeqLen=?), int64

    if split == 'train':
        set_seed(seed)
        indices = torch.randint(input_ids.size(-1) - seqlen + 1, [n_samples]).tolist()
    else:
        # test: ignore n_samples and load all
        indices = range(0, input_ids.size(-1) - seqlen + 1, seqlen)

    dataloader = torch.cat([input_ids[:, i: i + seqlen] for i in indices], dim=0)  # (N, SeqLen), int64
    return dataloader


def get_ptb(split: str, seqlen: int, n_samples: int, model_path: str, seed: int) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=False)
    dataset = load_dataset(path='ptb_text_only', name='penn_treebank', split=split, trust_remote_code=True)
    input_ids = tokenizer('\n\n'.join(dataset['sentence']), return_tensors='pt').input_ids  # (N=1, SeqLen=?), int64

    if split == 'train':
        set_seed(seed)
        indices = torch.randint(input_ids.size(-1) - seqlen + 1, [n_samples]).tolist()
    else:
        # validation or test: ignore n_samples and load all
        indices = range(0, input_ids.size(-1) - seqlen + 1, seqlen)

    dataloader = torch.cat([input_ids[:, i: i + seqlen] for i in indices], dim=0)  # (N, SeqLen), int64
    return dataloader


def get_c4(split: str, seqlen: int, n_samples: int, model_path: str, seed: int, new: bool) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=False)
    if split == 'train':
        dataset = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
        )
    else:
        dataset = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
        )

    if split == 'train' or not new:
        set_seed(seed)
        dataloader = []
        for _ in range(n_samples):
            while True:
                di = random.randint(0, len(dataset) - 1)
                input_ids = tokenizer(dataset[di]['text'], return_tensors='pt').input_ids  # (N=1, SeqLen=?), int64
                if input_ids.size(-1) >= seqlen:
                    break
            i = random.randint(0, input_ids.size(-1) - seqlen)
            dataloader.append(input_ids[:, i: i + seqlen])
        dataloader = torch.cat(dataloader, dim=0)
    else:
        input_ids = tokenizer(' '.join(dataset[:1100]['text']), return_tensors='pt').input_ids[:, :seqlen * n_samples]
        dataloader = torch.cat([
            input_ids[:, i: i + seqlen] for i in range(0, input_ids.size(-1) - seqlen + 1, seqlen)
        ], dim=0)
    return dataloader  # (N, SeqLen), int64


def get_mmlu(split: str, seqlen: int, n_samples: int, model_path: str, seed: int) -> torch.Tensor:
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, use_fast=False)
    dataset = load_dataset(path='cais/mmlu', name='all', split=split, trust_remote_code=True)
    # name: ['all', 'abstract_algebra', 'anatomy', 'astr...tudies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    # split: ['auxiliary_train', 'test', 'validation', 'dev']
    # features: ['question', 'subject', 'choices', 'answer']
    texts = [sample['question'] + ' ' + sample['choices'][sample['answer']] for sample in dataset]
    input_ids = tokenizer('\n\n'.join(texts), return_tensors='pt').input_ids  # (N=1, SeqLen=?), int64
    set_seed(seed)
    indices = torch.randint(input_ids.size(-1) - seqlen + 1, [n_samples]).tolist()
    dataloader = torch.cat([input_ids[:, i: i + seqlen] for i in indices], dim=0)  # (N, SeqLen), int64
    return dataloader
