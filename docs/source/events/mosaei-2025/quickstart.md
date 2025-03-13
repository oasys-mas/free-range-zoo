# Quickstart

---

## Overview 

Here we show a brief introduction on how to use a free-range-zoo environment, and how to use our baselines. For further detail see [Basic Usage](https://oasys-mas.github.io/free-range-zoo/introduction/basic_usage.html) and [Logging](https://oasys-mas.github.io/free-range-zoo/introduction/logging.html).

Here we use [Rideshare](https://oasys-mas.github.io/free-range-zoo/environments/rideshare/index.html) as a example domain, and we use `Random` and `noop` as example agents. We provide the full script of this tutorial at the bottom of this page for convenient copy and pasting. 

## Step #1: Environment Configurations

Configurations for each environment can be found at this 
[Kaggle link](https://www.kaggle.com/datasets/picklecat/moasei-aamas-2025-competition-configurations).
Configurations must be loaded using `pickle`. An example loading is shown below.

```python
import pickle

configuration = pickle.load(<path to configuration>)
```

## Step #2: Environment Creation

Now we create our environment giving it the configuration, and a number of parallel environments to create. Here `log_directory` is the location of a empty or nonexistant directory which environment logs will be saved. If not given (or if None) automatic logging will not occur.

```py
from free_range_zoo.envs import rideshare_v0
env = rideshare_v0.parallel_env(
    max_steps = 100,
    parallel_envs = 1,
    configuration = rideshare_configuration,
    device=torch.device('cpu'),
    log_directory = "test_logging"
)
```

Our baselines use the `action_mapping_wrapper`. This modifies the output `observation` to be a uple `space.Dict -> (space.Dict, space.Dict)`. The second dictionary, `t_mapping` allow us to identify which task is associated with which observation in cases where all agents do not observe the same tasks. Like with accepted passengers in rideshare.

```py
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
env = action_mapping_wrapper_v0(env)
```

## step #3: Environment Step

Now we can create our baseline agents and execute our policy. Here each `agent` must perform `observe` before each `act` which stores and process the prior observation. 

```py
from free_range_zoo.envs.rideshare.baselines import NoopBaseline, RandomBaseline

agents = {
    env.agents[0]: NoopBaseline(agent_name = "agent_0", parallel_envs = 1),
    env.agents[1]: RandomBaseline(agent_name = "agent_1", parallel_envs = 1)
}

while not torch.all(env.finished):
    
    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name][0])  # Policy observation 

    agent_actions = {
            agent_name:agents[agent_name].act(action_space = env.action_space(agent_name))
        for agent_name in env.agents
    }  # Policy action determination here

    observations, rewards, terminations, truncations, infos = env.step(agent_actions)

env.close()
```

Now you should see the directory `test_logging` with `test_logging/0.csv` where the number indicates the `parallel_env` index.

## Full Quickstart Script
```py
from free_range_zoo.envs import rideshare_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
import torch
import pickle

rideshare_configuration = pickle.load(<path to configuration>)

env = rideshare_v0.parallel_env(
    max_steps = 100,
    parallel_envs = 1,
    configuration = rideshare_configuration,
    device=torch.device('cpu'),
    log_directory = "test_logging"
)
env.reset()
env = action_mapping_wrapper_v0(env)
observations, infos = env.reset()

from free_range_zoo.envs.rideshare.baselines import NoopBaseline, RandomBaseline

agents = {
    env.agents[0]: NoopBaseline(agent_name = "agent_0", parallel_envs = 1),
    env.agents[1]: RandomBaseline(agent_name = "agent_1", parallel_envs = 1)
}

while not torch.all(env.finished):
    
    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name][0])  # Policy observation 

    agent_actions = {
            agent_name:agents[agent_name].act(action_space = env.action_space(agent_name))
        for agent_name in env.agents
    }  # Policy action determination here

    observations, rewards, terminations, truncations, infos = env.step(agent_actions)

env.close()
```
