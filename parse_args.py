import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='gptq-dynamic args')

    parser.add_argument(
        '--model-dir', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--seqlen', type=int, default=2048,
    )
    parser.add_argument(
        '--data-train-set', type=str, choices=['wikitext2', 'ptb', 'c4'], default='c4',
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--data-train-n-samples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--data-new-eval', type=str2bool, default=True,
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--data-seed', type=int, default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--data-cache-dir', type=str, default='./cache/datasets'
    )
    parser.add_argument(
        '--do-quant', type=str2bool, default=True,
    )
    parser.add_argument(
        '--save-model-path', type=str, default='./outputs/results.pkl'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
    )
    parser.add_argument(
        '--gpu-id', type=int, default=0,
    )

    args = parser.parse_args()
    return args
