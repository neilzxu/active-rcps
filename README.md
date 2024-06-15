# Code for "Active, anytime-valid risk controlling prediction sets"

Use `python=3.8.3` and `conda`

## Installation instructions

`conda install env -f environment.yml`
`pip install -r requirements.txt`

## Running experiments

Simulations: `python simulation.py --processes <processes> --out_dir <output directory of results>`

Imagenet experiments: `python imagenet_v2.py --processes <processes> --out_dir <output directory of results>`

Visualizing the results of the experiments: `python visualize.py --result_dir <directory of experiment results> --out_dir <directory to save figures to>`
