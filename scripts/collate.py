from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm


def append_arrays(npz_filepath, X, Y, Z):
    npz = np.load(npz_filepath)

    X.append(npz["X"])
    Y.append(npz["Y"])
    Z.append(npz["H"])

    return X, Y, Z


def pad(x, length):
    return np.pad(x, [(0, length - len(x)), (0, 0)], constant_values=np.nan)


def main():
    parser = ArgumentParser()
    parser.add_argument("parent_dir", type=Path)
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    X_full = []
    Y_full = []
    H_full = []

    X_prune = []
    Y_prune = []
    H_prune = []

    missing_reps = []
    for p in tqdm(args.parent_dir.iterdir(), "Collating", unit="rep"):
        if not p.is_dir():
            continue

        if not (args.parent_dir / "log.txt").exists():
            shutil.move(p / "log.txt", args.parent_dir / "log.txt")

        try:
            X_full, Y_full, H_full = append_arrays(p / "full.npz", X_full, Y_full, H_full)
            X_prune, Y_prune, H_prune = append_arrays(p / "prune.npz", X_prune, Y_prune, H_prune)
        except FileNotFoundError:
            missing_reps.append(p.stem.split("-")[-1])

    X_full = np.array(X_full)
    Y_full = np.array(Y_full).squeeze(-1)
    H_full = np.array(H_full)

    length = max(len(A) for A in X_prune)

    X_prune = np.array([pad(X, length) for X in X_prune])
    Y_prune = np.array([pad(Y, length) for Y in Y_prune]).squeeze(-1)
    H_prune = np.array(H_prune)

    print("X shape: ", X_prune.shape)
    print("Y shape: ", Y_prune.shape)
    print(f"Missing reps: {','.join(missing_reps)}")

    for filename, A_f, A_p in [
        ("X.npz", X_full, X_prune),
        ("Y.npz", Y_full, Y_prune),
        ("H.npz", H_full, H_prune),
    ]:
        np.savez_compressed(args.parent_dir / filename, FULL=A_f, PRUNE=A_p)

    if args.clean:
        for p in tqdm(args.parent_dir.iterdir(), "Cleaning", unit="rep"):
            if not p.is_dir():
                continue
            shutil.rmtree(p)


if __name__ == "__main__":
    main()
