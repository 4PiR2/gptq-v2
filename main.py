import logging
import sys
import time

import torch

from data_utils import get_dataloader
from llama import get_llama, quantize_llama, evaluate_llama
from parse_args import parse_args


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def main() -> None:
    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)
    logging.info(' '.join(sys.argv))
    args = parse_args()
    logging.info(args)

    model = get_llama(args.model_dir)
    model.eval()
    model_save_path = args.save_model_path
    device: torch.device = torch.device(f'cuda:{args.gpu_id}')

    encodings_train = get_dataloader(
        name=args.data_train_set, split='train', seqlen=args.seqlen, n_samples=args.data_train_n_samples,
        model_path=args.model_dir, seed=args.data_seed, cache_dir=args.data_cache_dir,
    )

    if args.do_quant:
        tick = time.time()
        results = quantize_llama(model=model, encodings=encodings_train, device=device, batch_size=args.batch_size)
        logging.info(f'finished quantizing in {time.time() - tick:.2f} s')

        if model_save_path:
            torch.save(results, model_save_path)

    dataset_names = ['train', 'wikitext2', f'c4{"-new" if args.data_new_eval else ""}', 'mmlu']

    for dataset_name in dataset_names:
        if dataset_name == 'train':
            encodings = encodings_train
        else:
            encodings = get_dataloader(
                name=dataset_name, split='test', seqlen=args.seqlen, model_path=args.model_dir, seed=args.data_seed,
                cache_dir=args.data_cache_dir,
            )
        logging.info(f'evaluating {dataset_name}')
        ppl = evaluate_llama(model=model, encodings=encodings, device=device, batch_size=args.batch_size)
        logging.info(f'ppl: {ppl.item():.4f}')


if __name__ == '__main__':
    main()
