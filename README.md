# Trajectory-Aware Eligibility Traces

This repository contains code for performing temporal credit assignment in reinforcement learning using *trajectory-aware* eligibility traces ("etraces").


## Setup

The code requires Python 3.6+ to run.
To get started, `cd` into the root directory and use `pip` to install the required dependencies.

```
cd trajectory-aware-etraces/
pip install -r requirements.txt
```


## Reproducing the Paper's Experiments

The scripts needed for reproducing the experiments in the paper are contained in `trajectory_aware_etraces/experiments/control`.
From the root directory,

1. `cd trajectory_aware_etraces/experiments/control`
1. (Optional) Edit `config.yml` to change the environment, algorithms, or any other experiment settings.

> **Note:** The default number of trials for all experiments is 1000, which may take a long time to run.
You may want to start with a smaller value (e.g., 5 or 10) when experimenting for faster iteration.

3. `python run_experiments.py` to generate the data (saved in `data/` by default).
1. (Optional) `python grid_search.py` to print the AUC (mean and 95% confidence interval) for all of the tested hyperparameter combinations.
    - Identify the best α-value for each λ-value.
    - Edit the `lambda_sweep_alphas` key in `config.yml` to set the identified values for each method.
    (The default values were used to generate the plots in the paper.)
1. `python lambda_sweep.py` to generate the λ-sweep plot.
1. `python learning_curves.py` to generate the learning curves.


## Counterexamples to Convergence

Violating Condition 5.1 can sometimes cause divergence, as we demonstrated in the paper with two counterexamples.
The code for these counterexamples, which calculates the Z matrix and its norm for the specified hyperparameter settings, can be run from the root directory:

**Counterexample 5.7** Off-Policy Truncated IS:
```
python counterexamples/offpolicy_truncated_is.py
```

**Counterexample 5.8** On-Policy Binary Traces:
```
python counterexamples/onpolicy_binary.py
```
