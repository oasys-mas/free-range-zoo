# MOSAEI 2025 - AAMAS
```{toctree}
:caption: MOSAEI 2025 - AAMAS
:hidden:

quickstart
environment_initialization
evaluation
```

Hello welcome to the MOASEI 2025 docs! MOASEI is a competition to see which policies can perform the best in open multi-agent systems. See competition details on the [MOASEI 2025 website](https://oasys-mas.github.io/moasei.html). Here we will discuss expectations on submissions, and the evaluation procedure.

Your objective is to construct a `Agent` class for your selected track which will perform the best across all **environment configurations** that we provide here [Kaggle](https://www.kaggle.com/datasets/picklecat/moasei-aamas-2025-competition-configurations). On that kaggle there are 3 configurations for each track (domain). Wildfire (DW) and Cyber Security (CS) have stochastic transition functions which are seeded. We will run the **one submitted policy for that track across all configurations shown, not shown configurations, across multiple seeds.** Policies earn points according to which place they score in each configuration. Policies are awarded `n-(k-1)` points for `n` many participating policies and `kth` place. The policy with highest points in the track wins that track.

> Rideshare's seeding is "baked in" to the configuration files, so we provide `20` configuration files for each configuration DR1, DR2, and DR3. These configuration files are seperate seeds for stochastics. **Resetting Rideshare will not stochastically change it unlike Wildfire and Cybersecurity.**

## Instructions

We recommend you follow the [installation guide](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html) to install free-range-zoo, then run the full [quickstart](https://oasys-mas.github.io/free-range-zoo/events/mosaei-2025/quickstart.html) script to verify your installation, and see the [basic usage guide](https://oasys-mas.github.io/free-range-zoo/introduction/basic_usage.html) to see a example of making an `Agent`.

## Submission

You must submit the following:

1. The source code of your `Agent` class with a list of all dependencies.
2. The source code used to train/update your `Agent`.
3. A modified version of the code shown in [evaluation](https://oasys-mas.github.io/free-range-zoo/events/mosaei-2025/evaluation.html) which initializes, loads, and evaluates your model.
4. The learned weights of your `Agent`.