# Environment Initialization

Hello welcome to the MOASEI 2025 docs! MOASEI is a competition to see which policies can perform the best in open multi-agent systems. See competition details on the [MOASEI 2025 website](https://oasys-mas.github.io/moasei.html). Here we will discuss expectations on submissions, and the evaluation procedure.

Your objective is to construct a `Agent` class for your selected track which will perform the best across all **environment configurations** that we provide here [Kaggle](https://www.kaggle.com/datasets/picklecat/moasei-aamas-2025-competition-configurations). Each of these configurations consistute one **seed** for that configuration. The **seed** controls the openness present in the environment. Your code will be evaluated on the shown **configurations**, on the shown **seeds**, and on **seeds** we have not revealed. The victor will be the policy that earns the highest total episodic rewards across all tested **seeds** and **configurations** within its track.

## Instructions

We recommend you follow the [installation guide](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html) to install free-range-zoo, then run the full [quickstart](https://oasys-mas.github.io/free-range-zoo/introduction/quickstart.html) script to verify your installation, and see the [basic usage guide](https://oasys-mas.github.io/free-range-zoo/introduction/basic_usage.html) to see a example of making an `Agent`.


## Submission

You must submit the following:

1. The source code of your `Agent` class with a list of all dependencies.
2. The source code used to train/update your `Agent`.
2. The learned weights of your `Agent`.


## Evaluation

The competition will be operating on specific environment initialization parameters. The initializations for each 
environment are detailed below.


### Wildfire
```python
env = wildfire_v0.parallel_env(
    parallel_envs=args.parallel_envs,
    max_steps=args.steps,
    configuration=configuration,
    device=device,
    buffer_size=args.buffer_size,
    show_bad_actions=False,
    observe_other_power=False,
    observe_other_suppressant=False,
)
```

### Rideshare
```python
env = rideshare_v0.parallel_env(
    parallel_envs=args.parallel_envs,
    max_steps=args.steps,
    configuration=configuration,
    device=device,
)
```

### Cybersecurity
```python
env = cybersecurity_v0.parallel_env(
    parallel_envs=args.parallel_envs,
    max_steps=args.steps,
    configuration=configuration,
    device=device,
    buffer_size=args.buffer_size,
    show_bad_actions=False,
)
```

### Environment Configurations

Configurations for each environment can be found at this 
[Kaggle link](https://www.kaggle.com/datasets/picklecat/moasei-aamas-2025-competition-configurations).
Configurations must be loaded using `pickle`. An example loading is shown below.

```python
import pickle

configuration = pickle.load(<path to configuration>)
```

This configuration can then be directly input to the environment configuration parameter of its respective domain.
