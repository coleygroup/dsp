python scripts/figures.py michalewicz ~/active-projects/boip/runs/all-objs/michalewicz ~/active-projects/boip/runs/michalewicz-gamma -o figures/github/michal-figure.png 
python scripts/figures.py combo ~/active-projects/boip/runs/all-objs/{beale,branin} -o figures/github/combo.png
python scripts/figures.py regret ~/active-projects/boip/runs/all-objs/michalewicz michalewicz -o figures/github/mich-regret.png
python scripts/figures.py gamma-perf ~/active-projects/boip/runs/michalewicz-gamma michalewicz -o figures/github/mich-gamma-perf.png