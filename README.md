# boip
**B**ayesian **O**ptimization with **I**nput space **P**runing

<!-- FIGURE HERE -->

# Overview
This repository contains code for replicating the data and figures of the synthetic test function experiments from the paper **INSERT_PAPER_NAME_HERE**

# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Testing](#testing)
- [Running BOIP](#running-boip)
- [Reproducing Data](#reproducing-data)
- [Citation](#citation)

# Setup

## Requirements
- `python==3.8`
- `botorch`, `gpytorch`, `pytorch`, `numpy`, `tqdm`
- **(figures)** `matplotlib`, `seaborn`
- **(testing)** `pytest`

## Installation
1. `conda env create -f enviroment.yml`
1. `pip install -e .`

# Testing

## Unit Tests
If BOIP was installed properly, all unit tests should pass. To run them:
1. install pytest: `pip install pytest`
1. run the tests: `pytest`

## Integration Testing
To perform a sample run of BOIP, run `boip --smoke-test`. This should generate output to your terminal containing the run parameters and perform 3 total runs. It will produce one folder, `smoke-test`, containing a single file, `out.npz`. This file should have 3 keys: `"X"`, `"Y"`, and `"H"`. If anything fails, check your installation and the unit tests first!

# Running BOIP

BOIP is run via the command line like so:
```
usage: boip [-h] [-o OBJECTIVE] [-c NUM_CHOICES] [-N N] [-q BATCH_SIZE] [-T T] [-R REPEATS]
            [-ds DISCRETIZATION_SEED] [-p PROB] [-a ALPHA] [--output-dir OUTPUT_DIR] [--smoke-test] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -o OBJECTIVE, --objective OBJECTIVE
  -c NUM_CHOICES, --num-choices NUM_CHOICES
                        the number of points with which to discretize the objective function
  -N N                  the number of initialization points
  -q BATCH_SIZE, --batch-size BATCH_SIZE
  -T T                  the number iterations to perform optimization
  -R REPEATS, --repeats REPEATS
                        the number of repetitions to perform. If not specified, then collate.py must be run afterwards for further analysis.
  -ds DISCRETIZATION_SEED, --discretization-seed DISCRETIZATION_SEED
                        the random seed to use for discrete landscapes
  -p PROB, --prob PROB  the minimum probability that a point is a hit for it to be retained
  -a ALPHA, --alpha ALPHA
                        the amount by which to scale the uncertainty estimates
  --output-dir OUTPUT_DIR
                        the directory under which to save the outputs
  --smoke-test
  -v, --verbose
```

The output directory will have a log file containing the parameters of the given and 3 NPZ files. If no value for `R` was provided, then the 3 files will `full.npz`, `prune.npz`, and `reacq.npz`, each with keys `"X"`, `"Y"`, and `"H"`. The `X` array corresponds to the points sampled (in order) for a given optimization setting: no pruning, pruning + no reacquisition, or pruning + reacquisition. The `Y` array is parallel to the `X` array and corresponds to the objective values of the given point. The `H` array is an array of shape `N x 2` and contains the iteration of when the given point was either acquired (0th column) or pruned (1st column.) Note that in the case of reacquisition, the 0th column of the `H` array represents only the *most recent* iteration of when that point was acquired.

If a value for `R` was provided, then the files will be inverted: there will be `X.npz`, `Y.npz`, and `H.npz`, each containing keys `"FULL"`, `"PRUNE"`, and `"REACQ"`. Each array is equivalent calling `np.stack` on the corresponding arrays from multiple, individual runs.

# Reproducing Data

## Experiments

Experiments for each objective were run like so:
```
boip -o OBJECTIVE -c 10000 -ds 42 -N 10 -T 200 -q 10 -p 0.025
```
the `--output-dir` argument for each run was of the form `path/to/OBJECTIVE/rep-I`, where `I` is the number of the given repetition. 100 repititions were performed for each run (using SLURM to maintain sanity.)

After each set of runs was complete, the runs were collated:
```
python scripts/collate.py --parent-dir path/to/OBJECTIVE
```

## Figures

See the [figures noteboook](notebooks/figs.ipynb) for details

# Citation

If you found this repository or its ideas even remotely useful in the course of your own work, you can cite it as follows:

**COMING SOON**