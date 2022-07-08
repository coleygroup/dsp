#!/bin/bash

DIR=${1:-./figures}
FMT=${2:-png}

python scripts/figures.py michalewicz ./data/all-objs/michalewicz ./data/michalewicz-gamma -o $DIR/michal-figure.${FMT} 
python scripts/figures.py combo ./data/all-objs/* -o $DIR/combo.${FMT}
python scripts/figures.py gamma ./data/michalewicz-gamma michalewicz -o $DIR/mich-gamma.${FMT}
python scripts/figures.py prob ./data/michalewicz-prob michalewicz -o $DIR/mich-prob.${FMT}
python scripts/figures.py perf ./data/michalewicz-obs michalewicz -o $DIR/mich-perf-obs.${FMT}
python scripts/figures.py perf ./data/all-objs/michalewicz michalewicz --all-traces -o $DIR/mich-perf-all.${FMT}
python scripts/figures.py fpr ./data/all-objs/michalewicz michalewicz --all-traces -o $DIR/mich-fpr-all.${FMT}