import argparse
import os
import pickle5 as pickle

import numpy as np


def load_imagenet_torch_preds(save_dir):
    names = os.listdir(save_dir)
    parts = len(names) - 1

    score_parts = []
    for i in range(parts):
        with open(f'{save_dir}/scores_{i}.pkl', 'rb') as in_f:
            score_parts.append(pickle.load(in_f))
    with open(f'{save_dir}/labels.pkl', 'rb') as in_f:
        labels = pickle.load(in_f)

    return np.concatenate(score_parts), labels


def save_imagenet_torch_preds(file_path, t_path, out_dir, parts=2):
    with open(args.file_path, 'rb') as in_f:
        torch_data = pickle.load(in_f)
    logits, labels = torch_data.tensors
    if t_path is not None:
        with open(args.t_path, 'rb') as in_f:
            T = pickle.load(in_f)
        scores = (logits.cpu() / T.cpu()).softmax(dim=1)
    else:
        scores = (logits.cpu()).softmax(dim=1)

    score_arr = scores.detach().numpy()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    indices = np.linspace(0, len(logits), num=parts + 1, dtype=int)
    for i in range(len(indices) - 1):
        with open(f'{args.out_dir}/scores_{i}.pkl', 'wb') as out_f:
            pickle.dump(score_arr[indices[i]:indices[i + 1]], out_f)
    with open(f'{args.out_dir}/labels.pkl', 'wb') as out_f:
        pickle.dump(labels.detach().cpu().numpy(), out_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Load cached predictions generated by torch models')
    parser.add_argument('--file_path',
                        required=True)  # option that takes a value
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--t_path', default=None)  # option that takes a value
    args = parser.parse_args()
    save_imagenet_torch_preds(args.file_path, args.t_path, args.out_dir)
