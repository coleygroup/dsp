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

def save_arrays(npz_filepath, full, prune, reacq):
    if npz_filepath.exists():
        npz = np.load(npz_filepath)
        full = np.concatenate((npz["FULL"], full))
        prune = np.concatenate((npz["PRUNE"], prune))
        reacq = np.concatenate((npz["REACQ"], reacq))

    np.savez_compressed(npz_filepath, FULL=full, PRUNE=prune, REACQ=reacq)

def main():
    parser = ArgumentParser()
    parser.add_argument("--parent-dir", type=Path)

    args = parser.parse_args()

    X_full = []
    Y_full = []
    H_full = []

    X_prune = []
    Y_prune = []
    H_prune = []

    X_reacq = []
    Y_reacq = []
    H_reacq = []

    for p in tqdm(args.parent_dir.iterdir(), "Collating", unit="rep"):
        if not p.is_dir():
            continue

        if not (args.parent_dir / "log.txt").exists():
            shutil.move(p / "log.txt", args.parent_dir / "log.txt")

        X_full, Y_full, H_full = append_arrays(p / "full.npz", X_full, Y_full, H_full)
        X_prune, Y_prune, H_prune = append_arrays(p / "prune.npz", X_prune, Y_prune, H_prune)
        X_reacq, Y_reacq, H_reacq = append_arrays(p / "reacq.npz", X_reacq, Y_reacq, H_reacq)

        # prune_npz = np.load(p / "prune.npz")
        # X_prune.append(prune_npz["X"])
        # Y_prune.append(prune_npz["Y"])
        # H_prune.append(prune_npz["H"])

        # reacq_npz = np.load(p / "reacq.npz")
        # X_reacq.append(reacq_npz["X"])
        # Y_reacq.append(reacq_npz["Y"])
        # H_reacq.append(reacq_npz["H"])

    X_full = np.array(X_full)
    Y_full = np.array(Y_full).squeeze(-1)
    H_full = np.array(H_full)

    X_prune = np.array(X_prune)
    Y_prune = np.array(Y_prune).squeeze(-1)
    H_prune = np.array(H_prune)

    X_reacq = np.array(X_reacq)
    Y_reacq = np.array(Y_reacq).squeeze(-1)
    H_reacq = np.array(H_reacq)

    save_arrays(args.parent_dir / 'X.npz', X_full, X_prune, X_reacq)
    save_arrays(args.parent_dir / 'Y.npz', Y_full, Y_prune, Y_reacq)
    save_arrays(args.parent_dir / 'H.npz', H_full, H_prune, H_reacq)
    
    for p in tqdm(args.parent_dir.iterdir(), "Cleaning", unit="rep"):
        if not p.is_dir():
            continue
        shutil.rmtree(p)

if __name__ == "__main__":
    main()