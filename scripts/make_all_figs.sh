python scripts/figures.py michalewicz ./data/all-objs/michalewicz ./data/michalewicz-gamma -o figures/github/michal-figure.png 
python scripts/figures.py combo ./data/all-objs/{beale,branin} -o figures/github/combo.png
python scripts/figures.py regret ./data/all-objs/michalewicz michalewicz -o figures/github/mich-regret.png
python scripts/figures.py gamma-perf ./data/michalewicz-gamma michalewicz -o figures/github/mich-gamma-perf.png