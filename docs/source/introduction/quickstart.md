# Quickstart

---

## Overview 

Here we show a brief introduction on how to use a free-range-zoo environment, and how to use our baselines. For further detail see [Basic Usage](https://oasys-mas.github.io/free-range-zoo/introduction/basic_usage.html) and [Logging](https://oasys-mas.github.io/free-range-zoo/introduction/logging.html).

Here we use [Rideshare](https://oasys-mas.github.io/free-range-zoo/environments/rideshare/index.html) as a example domain, and we use `Random` and `noop` as example agents. We provide the full script of this tutorial at the bottom of this page for convenient copy and pasting. 


## Step #1: Environment Configuration

Environments are located in `free_range_zoo/envs`. Each environment has a set of configuration dataclasses in `envs/<env_name>/env/structures/configuration.py`. 

For rideshare: `RewardConfiguration`, `PassengerConfiguration`, `AgentConfiguration`, and `RideshareConfiguration`. The `<env_name>Configuration` class holds the remaining configurations.

> Unless otherwise specified each setting applies to all parallel environments constructed with `parallel_envs`


For example we define a configuration below, 

```py
from free_range_zoo.envs.rideshare.env.structures.configuration import RewardConfiguration, PassengerConfiguration, AgentConfiguration, RideshareConfiguration
```

### Reward Function - `RewardConfiguration`

Here we define the reward function. In Rideshare there are various penalties that are applied for each action, making passenger wait, and going over the limit of passengers in one agent/car.

```py
reward_config = RewardConfiguration(
    pick_cost = -0.1,
    move_cost = -0.02,
    drop_cost = 0.0,
    noop_cost = -0.1,
    accept_cost = -0.1,
    pool_limit_cost = -4.0,

    use_pooling_rewards = False,
    use_variable_move_cost = False,
    use_waiting_costs = False,

    wait_limit = torch.tensor([5,3,3], dtype=int),
    long_wait_time = 5,
    general_wait_cost = -0.1,
    long_wait_cost = -0.5
)
```

### Task Schedule - `PassengerConfiguration`

Here we define the schedule of passengers entering the environment. This is a tensor of shape: `<#passengers, 7>`. Where we indicate `<entry timestep, which batch, y, x position, y,x destination, fare earned>`

```py
passenger_config = PassengerConfiguration(
    schedule = torch.tensor([[2,0, 0,0, 1,1, 3],])
)
```

The above creates one passenger in batch 0, which enters at timestep 2. 

### Agent Count/Position - `AgentConfiguration`

Here we define the starting position, number of agents, pooling limit, and method of travel for agents.


```py
agent_config = AgentConfiguration(
    start_positions = torch.tensor([
        [1,1],
        [2,2]
    ]),
    pool_limit = 2,

    use_diagonal_travel = False,
    use_fast_travel = True
)
```
With these settings, there are two agents, and these agents will "teleport" to their destination of choice on taking a action (still incurring cost for distance traveled).

### Step #2: Environment Config - `RideshareConfiguration`

Here we define the size of the grid, and we provide the other configurations.

```py
rideshare_config = RideshareConfiguration(
    grid_height = 5,
    grid_width = 5,

    reward_config = reward_config,
    passenger_config = passenger_config,
    agent_config = agent_config
)
```

## Step #3: Environment Creation

Now we create our environment giving it the configuration, and a number of parallel environments to create. Here `log_directory` is the location of a empty or nonexistant directory which environment logs will be saved. If not given (or if None) automatic logging will not occur.

```py
from free_range_zoo.envs import rideshare_v0
env = rideshare_v0.parallel_env(
    max_steps = 100,
    parallel_envs = 1,
    configuration = rideshare_config,
    device=torch.device('cpu'),
    log_directory = "test_logging"
)
```

Our baselines use the `action_mapping_wrapper`. This modifies the output `observation` to be a uple `space.Dict -> (space.Dict, space.Dict)`. The second dictionary, `t_mapping` allow us to identify which task is associated with which observation in cases where all agents do not observe the same tasks. Like with accepted passengers in rideshare.

```py
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
env = action_mapping_wrapper_v0(env)
```

## Step #4: Environment Step

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
from free_range_zoo.envs.rideshare.env.structures.configuration import RewardConfiguration, PassengerConfiguration, AgentConfiguration, RideshareConfiguration
from free_range_zoo.envs import rideshare_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
import torch

reward_config = RewardConfiguration(
    pick_cost = -0.1,
    move_cost = -0.02,
    drop_cost = 0.0,
    noop_cost = -0.1,
    accept_cost = -0.1,
    pool_limit_cost = -4.0,

    use_pooling_rewards = False,
    use_variable_move_cost = False,
    use_waiting_costs = False,

    wait_limit = torch.tensor([5,3,3], dtype=torch.int32),
    long_wait_time = 5,
    general_wait_cost = -0.1,
    long_wait_cost = -0.5
)

passenger_config = PassengerConfiguration(
    schedule = torch.tensor([[0,0, 0,0, 1,1, 3],], dtype=torch.int32)
)

agent_config = AgentConfiguration(
    start_positions = torch.tensor([
        [1,1],
        [2,2]
    ]),
    pool_limit = 2,

    use_diagonal_travel = False,
    use_fast_travel = True
)

rideshare_config = RideshareConfiguration(
    grid_height = 5,
    grid_width = 5,

    reward_config = reward_config,
    passenger_config = passenger_config,
    agent_config = agent_config
)


env = rideshare_v0.parallel_env(
    max_steps = 100,
    parallel_envs = 1,
    configuration = rideshare_config,
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