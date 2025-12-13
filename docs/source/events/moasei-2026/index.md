# MOASEI 2026 - AAMAS
```{toctree}
:caption: MOASEI 2026 - AAMAS
:hidden:

environment_initialization
evaluation
faq
quickstart_fire
quickstart_cyber
quickstart_ride
```

Hello welcome to the MOASEI 2026 docs! MOASEI is a competition to see which
policies can perform the best in open multi-agent systems. See competition
details on the [MOASEI 2026 website](https://oasys-mas.github.io/moasei.html).
Here we will discuss expectations on submissions, and the evaluation procedure.

Your objective is to construct a `Agent` class for your selected track which
will perform the best across all **environment configurations** that we provide
here
[Kaggle](https://www.kaggle.com/datasets/8aa1179938850fe46c50ad86c3aa526765963191dd18c4a13ce53211b2f7ca36).
On that kaggle there are 3 configurations for each track (domain). Wildfire
(DW) and Cyber Security (CS) have stochastic transition functions which are
seeded. We will run the **one submitted policy for that track across all
configurations shown, not shown configurations, across multiple seeds.**
Policies earn points according to which place they score in each configuration.
Policies are awarded `n-(k-1)` points for `n` many participating policies and
`kth` place. The policy with highest points in the track wins that track.

## Instructions

We recommend you follow the [installation
guide](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html)
to install free-range-zoo, then run one of the full
[quickstart](https://oasys-mas.github.io/free-range-zoo/events/moasei-2025/quickstart_ride.html)
(rideshare here) scripts to verify your installation, and see the [basic usage
guide](https://oasys-mas.github.io/free-range-zoo/introduction/basic_usage.html)
to see a example of making an `Agent`.

## Submission

You must submit the following:

1. The source code of your `Agent` class with a list of all dependencies.
2. The source code used to train/update your `Agent`.
3. A modified version of the code shown in
   [evaluation](https://oasys-mas.github.io/free-range-zoo/events/moasei-2025/evaluation.html)
   which initializes, loads, and evaluates your model.
4. The learned weights of your `Agent`.
