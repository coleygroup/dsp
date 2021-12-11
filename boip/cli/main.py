import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import ray
from tqdm import tqdm

import boip
from boip.cli.args import parse_args

def stack_results(results: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ys, Ss = zip(*results)

    X = np.stack([X.cpu().numpy() for X in Xs]).astype("f")
    Y = np.stack([Y.flatten().cpu().numpy() for Y in Ys]).astype("f")
    S = np.stack(Ss).astype("I")
    # T = np.stack(Ts)

    return X, Y, S

def main():
    args = parse_args()

    try:
        if "redis_password" in os.environ:
            ray.init(
                address=os.environ["ip_head"],
                _node_ip_address=os.environ["ip_head"].split(":")[0],
                _redis_password=os.environ["redis_password"],
            )
        else:
            ray.init(address="auto")
    except ConnectionError:
        ray.init()
    except PermissionError:
        print("Failed to create a temporary directory for ray")
        raise
    print(f"Connected to ray cluster with resources: {ray.cluster_resources()}")

    obj = boip.build_objective(args.objective)
    choices = boip.discretize(obj, args.num_choices, args.discretization_seed)
    
    if args.smoke_test:
        results = [boip.optimize(
            obj, args.N, args.T, choices, args.batch_size, True, args.N, args.prob, verbose=i==0
        ) for i in tqdm(range(3), "smoke test")]
        X, Y, S = stack_results(results)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / 'test.npz', X=X, Y=Y, S=S)
        exit()

    optimize = ray.remote(boip.optimize)

    refs_full = [
        optimize.remote(obj, args.N, args.T, choices, args.batch_size, False)
        for _ in range(args.repeats)
    ]
    refs_prune = [
        optimize.remote(obj, args.N, args.T, choices, args.batch_size, True, args.N, args.prob)
        for _ in range(args.repeats)
    ]

    trials_full = [ray.get(r) for r in tqdm(refs_full, desc='full', unit='rep')]
    trials_prune = [ray.get(r) for r in tqdm(refs_prune, desc='pruned', unit='rep')]
    Xs, Ys, Ss = zip(
        *(stack_results(trials) for trials in [trials_full, trials_prune])
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
    # np.savez(output_dir / 'T.npz', **dict(zip(labels, Ts)))

if __name__ == '__main__':
    main()