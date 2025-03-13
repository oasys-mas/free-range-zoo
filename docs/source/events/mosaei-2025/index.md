# MOSAEI 2025 - AAMAS
```{toctree}
:caption: MOSAEI 2025 - AAMAS
:hidden:

quickstart
environment_initialization
evaluation
```

Hello welcome to the MOASEI 2025 docs! MOASEI is a competition to see which policies can perform the best in open multi-agent systems. See competition details on the [MOASEI 2025 website](https://oasys-mas.github.io/moasei.html). Here we will discuss expectations on submissions, and the evaluation procedure.

Your objective is to construct a `Agent` class for your selected track which will perform the best across all **environment configurations** that we provide here [Kaggle](https://www.kaggle.com/datasets/picklecat/moasei-aamas-2025-competition-configurations). Each of these configurations consistute one **seed** for that configuration. The **seed** controls the openness present in the environment. Your code will be evaluated on the shown **configurations**, on the shown **seeds**, and on **seeds** we have not revealed. The victor will be the policy that earns the highest total episodic rewards across all tested **seeds** and **configurations** within its track.

## Instructions

We recommend you follow the [installation guide](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html) to install free-range-zoo, then run the full [quickstart](https://oasys-mas.github.io/free-range-zoo/events/mosaei-2025/quickstart.html) script to verify your installation, and see the [basic usage guide](https://oasys-mas.github.io/free-range-zoo/introduction/basic_usage.html) to see a example of making an `Agent`.

## Submission

You must submit the following:

1. The source code of your `Agent` class with a list of all dependencies.
2. The source code used to train/update your `Agent`.
3. The learned weights of your `Agent`.