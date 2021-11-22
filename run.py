from argparse import ArgumentParser
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import ray
from tqdm import tqdm

from boip import objectives
from boip.optimize import optimize

try:
    ray.init(address='auto')
except ConnectionError:
    ray.init()
except PermissionError:
    print('Failed to create a temporary directory for ray')
    raise


optimize = ray.remote(optimize)

def get_results_from_trials(trials: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ys, Ss, Ts = zip(*trials)

    X = np.stack([X.cpu().numpy() for X in Xs]).astype("f")
    Y = np.stack([Y.flatten().cpu().numpy() for Y in Ys]).astype("f")
    S = np.stack(Ss).astype("I")
    T = np.stack(Ts)

    return X, Y, S, T

def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--objective')
    parser.add_argument('-N', type=int, default=10, help='the number of initialization points')
    parser.add_argument('-T', type=int, default=100,
                        help='the number iterations to perform optimization')
    parser.add_argument('-R', '--repeats', type=int, default=5)
    parser.add_argument('-c', '--num-choices', type=int,
                        help='the number of points with which to discretize the objective function')
    parser.add_argument('-ds', '--discretization-seed', type=int,
                        help='the random seed to use for discrete landscapes')
    parser.add_argument('-k', type=int)
    parser.add_argument('--prob')
    parser.add_argument('--output-dir', help='the directory under which to save the outputs')

    args = parser.parse_args()

    obj = objectives.build_objective(args.objective)
    if args.num_choices is not None:
        choices = objectives.discretize(obj, args.num_choices, args.discretization_seed)
    else:
        choices = None

    refs_full = [
        optimize.remote(obj, args.N, args.T, choices, False) for _ in range(args.repeats)
    ]
    refs_prune = [
        optimize.remote(obj, args.N, args.T, choices, True, 1, 0.025) for _ in range(args.repeats)
    ]

    trials_full = [ray.get(r) for r in tqdm(refs_full, desc='full', unit='rep')]
    trials_prune = [ray.get(r) for r in tqdm(refs_prune, desc='pruned', unit='rep')]
    Xs, Ys, Ss, Ts = zip(
        *(get_results_from_trials(trials) for trials in [trials_full, trials_prune])
    )

    labels = ('FULL', 'PRUNE')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'log.txt', 'w') as fid:
        for k, v in sorted(vars(args).items()):
            fid.write(f'{k}: {v}\n')

    np.savez(output_dir / 'X.npz', **dict(zip(labels, Xs)))
    np.savez(output_dir / 'Y.npz', **dict(zip(labels, Ys)))
    np.savez(output_dir / 'S.npz', **dict(zip(labels, Ss)))
    np.savez(output_dir / 'T.npz', **dict(zip(labels, Ts)))

if __name__ == '__main__':
    main()