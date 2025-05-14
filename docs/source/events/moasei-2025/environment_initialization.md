# Environment Initialization

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
