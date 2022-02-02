from pathlib import Path
from typing import List, Tuple
from matplotlib import use

import numpy as np
from torch import Tensor
from tqdm import tqdm

import boip
from boip.cli.args import parse_args


def collate_results(
    results: List[Tuple[Tensor, Tensor, Tensor]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ys, Hs = zip(*results)

    X = np.stack([X.cpu().numpy() for X in Xs]).astype("f")
    Y = np.stack([Y.flatten().cpu().numpy() for Y in Ys]).astype("f")
    H = np.stack([H.cpu().numpy() for H in Hs]).astype("i4")

    return X, Y, H


def main():
    args = parse_args()
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print()

    if args.smoke_test:
        obj = boip.build_objective("michalewicz")
        choices = boip.discretize(obj, 10000, 42)
        results = [
            boip.optimize(
                obj, 10, 20, choices, 10, True, 10, 0.025, 1, 
                use_predicted_threshold=False, verbose=True
            )
            for _ in tqdm(range(3), "smoke test")
        ]
        X, Y, H = collate_results(results)

        Path("smoke-test").mkdir(parents=True, exist_ok=True)
        np.savez("smoke-test/out.npz", X=X, Y=Y, H=H)
        exit()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "log.txt", "w") as fid:
        for k, v in sorted(vars(args).items()):
            fid.write(f"{k}: {v}\n")

    obj = boip.build_objective(args.objective)
    choices = boip.discretize(obj, args.num_choices, args.discretization_seed)

    if args.repeats is None or args.repeats <= 1:
        full = boip.optimize(
            obj,
            args.N,
            args.T,
            choices,
            args.batch_size,
            False,
            init_mode=args.init_mode,
            verbose=args.verbose,
        )
        prune = boip.optimize(
            obj,
            args.N,
            args.T,
            choices,
            args.batch_size,
            True,
            args.k_or_threshold,
            args.prob,
            args.gamma,
            not args.use_observed_threshold,
            init_mode=args.init_mode,
            verbose=args.verbose,
        )
        # reacq = boip.optimize(
        #     obj,
        #     args.N,
        #     args.T,
        #     choices,
        #     args.batch_size,
        #     True,
        #     args.N,
        #     args.prob,
        #     args.alpha,
        #     False,
        #     verbose=args.verbose,
        # )

        keys = ("X", "Y", "H")
        np.savez_compressed(output_dir / "full.npz", **dict(zip(keys, full)))
        np.savez_compressed(output_dir / "prune.npz", **dict(zip(keys, prune)))
        # np.savez_compressed(output_dir / "reacq.npz", **dict(zip(keys, reacq)))

        exit()

    results_full = [
        boip.optimize(
            obj, args.N, args.T, choices, args.batch_size, False, verbose=args.verbose
        )
        for _ in tqdm(range(args.repeats), "full", unit="rep")
    ]
    results_prune = [
        boip.optimize(
            obj,
            args.N,
            args.T,
            choices,
            args.batch_size,
            True,
            args.k_or_threshold,
            args.prob,
            args.gamma,
            True,
            args.verbose,
        )
        for _ in tqdm(range(args.repeats), "pruning", unit="rep")
    ]
    # results_reacq = [
    #     boip.optimize(
    #         obj,
    #         args.N,
    #         args.T,
    #         choices,
    #         args.batch_size,
    #         True,
    #         args.N,
    #         args.prob,
    #         args.alpha,
    #         False,
    #         args.verbose,
    #     )
    #     for _ in tqdm(range(args.repeats), "reacquisition", unit="rep")
    # ]

    Xs, Ys, Hs = zip(
        *(collate_results(trials) for trials in [results_full, results_prune])
    )
    keys = ("FULL", "PRUNE")

    np.savez(output_dir / 'X.npz', **dict(zip(keys, Xs)))
    np.savez(output_dir / 'Y.npz', **dict(zip(keys, Ys)))
    np.savez_compressed(output_dir / 'H.npz', **dict(zip(keys, Hs)))

if __name__ == "__main__":
    main()
