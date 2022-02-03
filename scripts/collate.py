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
    return np.pad(x, ((0, length-len(x)), (0, 0)), constant_values=np.nan)

def pad_arrays(A, B):
    delta = (np.array(A.shape) - np.array(B.shape))[1]
    if delta > 0:
        return pad(B, delta)


def save_arrays(npz_filepath, full, prune):
    # if npz_filepath.exists():
    #     npz = np.load(npz_filepath)
    #     full = np.concatenate((npz["FULL"], full))
    #     prune = np.concatenate((npz["PRUNE"], prune))
    #     # reacq = np.concatenate((npz["REACQ"], reacq))

    np.savez_compressed(npz_filepath, FULL=full, PRUNE=prune)

def main():
    parser = ArgumentParser()
    parser.add_argument("--parent-dir", type=Path)
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    X_full = []
    Y_full = []
    H_full = []

    X_prune = []
    Y_prune = []
    H_prune = []

    missing_runs = []
    for p in tqdm(args.parent_dir.iterdir(), "Collating", unit="rep"):
        if not p.is_dir():
            continue

        if not (args.parent_dir / "log.txt").exists():
            shutil.move(p / "log.txt", args.parent_dir / "log.txt")

        try:
            X_full, Y_full, H_full = append_arrays(p / "full.npz", X_full, Y_full, H_full)
            X_prune, Y_prune, H_prune = append_arrays(p / "prune.npz", X_prune, Y_prune, H_prune)
        except FileNotFoundError:
            missing_runs.append(p)

    X_full = np.array(X_full)
    Y_full = np.array(Y_full).squeeze(-1)
    H_full = np.array(H_full)

    try:
        X_prune = np.array(X_prune)
        Y_prune = np.array(Y_prune).squeeze(-1)
    except ValueError:
        length = max(len(A) for A in X_prune)
        X_prune = np.array([pad(X, length) for X in X_prune])
        Y_prune = np.array([pad(Y, length) for Y in Y_prune]).squeeze(-1)
    H_prune = np.array(H_prune)

    print(X_prune.shape, Y_prune.shape)
    print(f"Missing runs: {[p.stem for p in missing_runs]}")

    # X_reacq = np.array(X_reacq)
    # Y_reacq = np.array(Y_reacq).squeeze(-1)
    # H_reacq = np.array(H_reacq)

    save_arrays(args.parent_dir / 'X.npz', X_full, X_prune)
    save_arrays(args.parent_dir / 'Y.npz', Y_full, Y_prune)
    save_arrays(args.parent_dir / 'H.npz', H_full, H_prune)
        
    if args.clean:
        for p in tqdm(args.parent_dir.iterdir(), "Cleaning", unit="rep"):
            if not p.is_dir():
                continue
            shutil.rmtree(p)

if __name__ == "__main__":
    main()