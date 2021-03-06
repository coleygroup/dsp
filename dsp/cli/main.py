from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch import Tensor
from tqdm import tqdm

import dsp
from dsp.cli.args import parse_args

ART = [
    "******************************************",
    "*     ___      ___      ___       ___    *",
    "*    | _ )    / _ \\    |_ _|     | _ \\   *",
    "*    | _ \\   | (_) |    | |      |  _/   *",
    "*    |___/    \\___/    |___|    _|_|_    *",
    '*  _|"""""| _|"""""| _|"""""| _| """ |   *',
    "*  \"`-0-0-' \"`-0-0-' \"`-0-0-' \"`-0-0-'   *",
    "******************************************",
]


def collate_results(
    results: List[Tuple[Tensor, Tensor, Tensor]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, Ys, Hs = zip(*results)

    X = np.stack([X.cpu().numpy() for X in Xs]).astype("f")
    Y = np.stack([Y.flatten().cpu().numpy() for Y in Ys]).astype("f")
    H = np.stack([H.cpu().numpy() for H in Hs]).astype("i4")

    return X, Y, H


def main():
    print("\n".join(ART))
    print("dsp will be run with the following arguments:")
    args = parse_args()
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print()

    if args.smoke_test:
        obj = dsp.build_objective("michalewicz")
        choices = dsp.discretize(obj, 10000, 42)
        results = [
            dsp.optimize(obj, 10, 20, choices, 10, True, 10, 0.025, gamma=2.0, verbose=True)
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

    obj = dsp.build_objective(args.objective)
    choices = dsp.discretize(obj, args.num_choices, args.discretization_seed)

    if args.repeats is None or args.repeats <= 1:
        full = dsp.optimize(
            obj, args.N, args.T, choices, args.batch_size, False, verbose=args.verbose
        )
        prune = dsp.optimize(
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
            verbose=args.verbose,
        )

        keys = ("X", "Y", "H")
        np.savez_compressed(output_dir / "full.npz", **dict(zip(keys, full)))
        np.savez_compressed(output_dir / "prune.npz", **dict(zip(keys, prune)))

        exit()

    results_full = [
        dsp.optimize(obj, args.N, args.T, choices, args.batch_size, False, verbose=args.verbose)
        for _ in tqdm(range(args.repeats), "full", unit="rep")
    ]
    results_prune = [
        dsp.optimize(
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

    Xs, Ys, Hs = zip(*(collate_results(trials) for trials in [results_full, results_prune]))
    keys = ("FULL", "PRUNE")

    np.savez(output_dir / "X.npz", **dict(zip(keys, Xs)))
    np.savez(output_dir / "Y.npz", **dict(zip(keys, Ys)))
    np.savez_compressed(output_dir / "H.npz", **dict(zip(keys, Hs)))


if __name__ == "__main__":
    main()
