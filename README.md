# Code for "Active, Anytime-valid Conformal Risk Control"

Use `python=3.8.3`.

## Installation instructions

`conda install env -f environment.yml`
`pip install -r requirements.txt`

## Simulations

- `onedbinomialtest.ipynb`: simulation for testing the Active CS
- `onedbinomialtestlesshypers.ipynb`: simulation for testing the Active CS and comparing them to using Cocob (no hyperparameters) for optimization
- `risk_control.ipynb`: another simulation for testing CS, this time for 10-d covariates where the relationship between X and Y is a logistic regression.

## Real real data experiments
The first is Imagenet experiments. This can be run simply by `python imagenet.py`.

`python imagenet_wealth.py` does not produce a CS, but rather tracks the wealth process for a single choice of $\beta$.

The second is COCO MS experiments. This can be run by `python coco.py --processes <# of processes>`, with any choice of processesors that are larger than 0.

Results for all these experiments can be visualized by executing the cells in `real_data.ipynb`.