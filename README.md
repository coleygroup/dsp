[//]: # (Badges)
[![CI](https://github.com/coleygroup/dsp/actions/workflows/CI.yaml/badge.svg)](https://github.com/coleygroup/dsp/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/coleygroup/dsp/branch/main/graph/badge.svg?token=DBBHSQLW8A)](https://codecov.io/gh/coleygroup/dsp)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]((https://github.com/psf/black))

# DSP
Bayesian Optimization with Active **D**esign **S**pace **P**runing

<!-- FIGURE HERE -->

# Overview
This repository contains code for replicating the data and figures of the synthetic test function experiments from the paper "Self-focusing virtual screening with active design space pruning"

# Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Testing](#testing)
- [Contributing](#contributing)
- [Running dsp](#running-dsp)
- [Reproducing Data](#reproducing-data)
- [Citation](#citation)

# Setup

## Requirements
- `python==3.8`
- `botorch`, `gpytorch`, `pytorch`, `numpy`, `scipy`, `tqdm`
- **(plotting)** `matplotlib`, `seaborn`
- **(testing)** `pytest`

## Installation
1. `conda env create -f  `[`enviroment.yml`](./environment.yml)

    *note:* you may need to edit this file to change to `pytorch` and `cudatoolkit` packages for your own system.
1. `pip install .`

# Testing

## Unit Tests
If DSP was installed properly, all unit tests should pass. To run them:
1. install the package with additional testing requirements: `pip install .[test]`
1. run the tests: `pytest`

## Integration Testing
To perform a sample run of DSP, run `dsp --smoke-test`. This should generate output to your terminal containing the run parameters and perform 3 total runs. It will produce one folder, `smoke-test`, containing a single file, `out.npz`. This file should have 3 keys: `"X"`, `"Y"`, and `"H"`. If anything fails, check your installation and the unit tests first!

# Contributing
See the [contribution guide](./CONTRIBUTING.md)

# Running DSP

DSP is run via the command line like so:
```
usage: dsp [-h] [-o {levy,beale,bukin,branin,camel,michalewicz,drop-wave}] [-c NUM_CHOICES] [-N N] [-q BATCH_SIZE] [-T T] [-R REPEATS]
            [-ds DISCRETIZATION_SEED] [-p PROB] [--k-or-threshold K_OR_THRESHOLD] [--use-observed-threshold]
            [-g GAMMA] [--output-dir OUTPUT_DIR] [--smoke-test] [--init-mode INIT_MODE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -o OBJECTIVE, --objective OBJECTIVE
                        the test function to use (case insensitive)
  -c NUM_CHOICES, --num-choices NUM_CHOICES
                        the number of points with which to discretize the objective function
  -N N                  the number of initialization points
  -q BATCH_SIZE, --batch-size BATCH_SIZE
  -T T                  the number of iterations to perform optimization
  -R REPEATS, --repeats REPEATS
                        the number of repetitions to perform. If not specified, then collate.py must be run afterwards for further analysis
  -ds DISCRETIZATION_SEED, --discretization-seed DISCRETIZATION_SEED
                        the random seed to use for discrete landscapes
  -p PROB, --prob PROB  the minimum hit probability needed to retain a given point during pruning
  --k-or-threshold K_OR_THRESHOLD
                        the rank of the predictions (int) or absolute threshold (float) to use when determing what constitutes a predicted hit
  --use-observed-threshold
                        if using rank-based hit thresholding, calculate the threshold from the k-th best observation, rather than the k-th best predicted mean
  -g GAMMA, --gamma GAMMA
                        the amount by which to scale the predicted variances
  --output-dir OUTPUT_DIR
                        the directory under which to save the outputs
  --smoke-test
  -v, --verbose
```
The output directory will have a log file containing the parameters of the given and 3 NPZ files. If no value for `R` was provided, then the 3 files will `full.npz`, `prune.npz`, and `reacq.npz`, each with keys `"X"`, `"Y"`, and `"H"`. The `X` array corresponds to the points sampled (in order) for a given optimization setting: no pruning, pruning + no reacquisition, or pruning + reacquisition. The `Y` array is parallel to the `X` array and corresponds to the objective values of the given point. The `H` array is an array of shape `N x 2` and contains the iteration of when the given point was either acquired (0th column) or pruned (1st column.) Note that in the case of reacquisition, the 0th column of the `H` array represents only the *most recent* iteration of when that point was acquired.

If a value for `R` was provided, then the files will be inverted: there will be `X.npz`, `Y.npz`, and `H.npz`, each containing keys `"FULL"`, `"PRUNE"`, and `"REACQ"`. Each array is equivalent calling `np.stack` on the corresponding arrays from multiple, individual runs.

# Reproducing Data

## Experiments
The three sets of experiments were run like so:

1. general regret plots:

`dsp -o OBJECTIVE -c 10000 -ds 42 -N 10 -T 200 -q 10 -p 0.025`

2. `gamma` sweep: 

`dsp -o michalewicz -c 10000 -ds 42 -N 10 -T 200 -q 10 -p 0.025 --gamma GAMMA`, where `GAMMA` is either `0.5`, `1.0`, or `2.0` (*note*: the `michalewicz` run from above is equivalent to setting `gamma` equal to `1.0`)

3. observed hit thresholding: 

`dsp -o michalewicz -c 10000 -ds 42 -N 10 -T 200 -q 10 -p 0.025  --use-observed-threshold`

the `--output-dir` argument for each run was of the form `path/to/OBJECTIVE/rep-R`, where `R` is the number of the given repetition. 100 repititions were performed for each run (using SLURM to maintain sanity.)

4. `p` sweep:

`dsp -o michalewicz -c 10000 -ds 42 -N 10 -T 200 -q 10 -p P`, where `P` is either `0.0125`, `0.025`, or `0.05` (*note*: the `michalewicz` run from above is equivalent to setting `p` equal to `0.025`)

## Processing and organizing data
After each set of runs was complete, the runs were collated: `python scripts/collate.py --parent-dir path/to/parent_dir`, where each individual individual run is stored under `parent_dir` and given a unique name.

You can optionally run this script with the `--clean` flag to consolidate your directory structure by deleting the individual run subdirectories (no information is lost as all runs are stored in the resulting array). However, you can't rerun the script with new data after `clean`ing, i.e., perform additional runs and stack them onto the `collate`d results.

To create the figures using the [figures script](scripts/figures.py), the processed data should be organized like the [data](./data) directory. That is, processed data should generally be organized under a directory with the name of the objective to which the data corresponds. The exception to this is `gamma` sweep data, which should all be organized under some grandparent directory, e.g., `gamma-sweep`, and then each directory should be the value of `gamma` to which the data corresponds.

## Figures
Figures used in the manuscript were generated via the [figures script](./scripts/figures.py). The script may be run like so:
```
usage: figures.py [-h] {michalewicz,combo,perf,gamma,prob_single,prob_triple,fpr} ...

optional arguments:
  -h, --help            show this help message and exit

figure:
  the figure you would like to make

  {michalewicz,combo,perf,gamma,prob_single,prob_triple,fpr}
    michalewicz         michalewicz multi-panel figure
    combo               surface+performance multi-panel figure
    perf                performance plot for a single objective
    gamma               gamma sweep multi-panel figure
    prob_single         probability sweep single-panel FPR figure
    prob_triple         probability sweep multi-panel performance+pruning figure
    fpr                 False pruning rate plot for a single objective
```
To see additional arguments needed for the corresponding figure, run the script with the desired `figure` followed by the `--help` flag. The data used to generate figures in the manuscript is located in the [data](./data) directory, and a sample script to generate a few of the figures is located [here](./scripts/make_all_figs.sh). Note that the script has a few variables hard-coded in: the design space space size, the discretization seed, the number of iterations, the batch size, and the number of iterations. If you run your experiments with non-default values (as defined above,) then you'll have to edit them in the script.

# Citation

If you found the code or the ideas in this repository even remotely useful in the course of your own work, you can cite it as follows:

**COMING SOON**