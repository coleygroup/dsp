from argparse import ArgumentParser
from pathlib import Path

import numpy as np

def load_arrays(*npz_files):
    fulls = []
    prunes = []
    for npz_file in npz_files:
        npz = np.load(npz_file)
        fulls.append(npz["FULL"])
        prunes.append(npz["PRUNE"])

    return fulls, prunes

def concat_prunes(prunes):
    n_evals = np.array([A.shape[1] for A in prunes])
    idx = np.argmax(n_evals)
    pad_width = [(0,0)] * prunes[0].ndim

    for i, A in enumerate(prunes):
        if i == idx:
            continue
        pad_width[1] = (0, n_evals[idx]-A.shape[1])
        prunes[i] = np.pad(A, pad_width, constant_values=np.nan)

    return np.concatenate(prunes, 0)

def main():
    parser = ArgumentParser()
    parser.add_argument("--npz-files", nargs="+")
    parser.add_argument("-o", "--output")
    parser.add_argument("-N", type=int)
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    fulls, prunes = load_arrays(*args.npz_files)

    full = np.concatenate(fulls, 0)[:args.N]
    prune = concat_prunes(prunes)[:args.N]

    print("Output shapes")
    print("Full: ", full.shape)
    print("Prune: ", prune.shape)
    
    if args.output:
        np.savez_compressed(args.output, FULL=full, PRUNE=prune)
    else:
        for i, npz_file in enumerate(args.npz_files):
            print(f"({i+1}) {npz_file}")
        idx = int(input("Select file to overwite (index): ")) - 1
        output = args.npz_files[idx]
        np.savez_compressed(output, FULL=full, PRUNE=prune)

if __name__ == "__main__":
    main()