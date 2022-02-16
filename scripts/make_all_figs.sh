#!/bin/bash

DIR=${1:-./figures}

python scripts/figures.py michalewicz ./data/all-objs/michalewicz ./data/michalewicz-gamma -o $DIR/michal-figure.png 
python scripts/figures.py combo ./data/all-objs/* -o $DIR/combo.png
python scripts/figures.py regret ./data/all-objs/michalewicz michalewicz -o $DIR/mich-regret.png
python scripts/figures.py gamma-perf ./data/michalewicz-gamma michalewicz -o $DIR/mich-gamma-perf.png