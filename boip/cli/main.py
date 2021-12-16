import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
# import ray
from tqdm import tqdm

import boip
from boip.cli.args import parse_args

def stack_results(results: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ys, Hs = zip(*results)

    X = np.stack([X.cpu().numpy() for X in Xs]).astype("f")
    Y = np.stack([Y.flatten().cpu().numpy() for Y in Ys]).astype("f")
    H = np.stack(Hs).astype("i2")

    return X, Y, H

def main():
    args = parse_args()

    # try:
    #     if "redis_password" in os.environ:
    #         ray.init(
    #             address=os.environ["ip_head"],
    #             _node_ip_address=os.environ["ip_head"].split(":")[0],
    #             _redis_password=os.environ["redis_password"],
    #         )
    #     else:
    #         ray.init(address="auto")
    # except ConnectionError:
    #     ray.init()
    # except PermissionError:
    #     print("Failed to create a temporary directory for ray")
    #     raise
    # print(f"Connected to ray cluster with resources: {ray.cluster_resources()}")

    if args.smoke_test:
        obj = boip.build_objective("michalewicz")
        choices = boip.discretize(obj, 10000, 42)
        results = [
            boip.optimize(
                obj, 10, 20, choices, 10, True, 10, 0.025, True
            )
            for _ in tqdm(range(3), "smoke test")
        ]
        X, Y, H = stack_results(results)

        Path("smoke-test").mkdir(parents=True, exist_ok=True)
        np.savez('smoke-test/out.npz', X=X, Y=Y, H=H)
        exit()

    obj = boip.build_objective(args.objective)
    choices = boip.discretize(obj, args.num_choices, args.discretization_seed)

    results_full = [
        boip.optimize(obj, args.N, args.T, choices, args.batch_size, False)
        for _ in tqdm(range(args.repeats), 'full', unit='rep')
    ]
    results_prune = [
        boip.optimize(obj, args.N, args.T, choices, args.batch_size, True, args.N, args.prob)
        for _ in tqdm(range(args.repeats), 'pruning', unit='rep')
    ]
    Xs, Ys, Hs = zip(
        *(stack_results(trials) for trials in [results_full, results_prune])
    )

    labels = ('FULL', 'PRUNE')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'log.txt', 'w') as fid:
        for k, v in sorted(vars(args).items()):
            fid.write(f'{k}: {v}\n')

    np.savez(output_dir / 'X.npz', **dict(zip(labels, Xs)))
    np.savez(output_dir / 'Y.npz', **dict(zip(labels, Ys)))
    np.savez(output_dir / 'H.npz', **dict(zip(labels, Hs)))

if __name__ == '__main__':
    main()