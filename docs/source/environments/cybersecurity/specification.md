---
autogenerated: true
lastpage: wildfire
---

# Specification

---

| Import             | `from free_range_zoo.envs import cybersecurity_v0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Actions            | Discrete & Stochastic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Observations       | Discrete and Partially Observed with Private Observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Parallel API       | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Manual Control     | No                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Agent Names        | [$attacker_0$, ..., $attacker_n$, $defender_0$, ..., $defender_n$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| # Agents           | [0, $n_{attackers}$ + $n_{defenders}$]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Action Shape       | ($envs$, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Action Values      | Attackers: [$attack_0$, ..., $attack_{tasks}$, $noop$ (-1)]<br>Defenders: [$move_0$, ..., $move_{tasks}$, $noop$ (-1), $patch$ (-2), $monitor$ (-3)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Observation Shape  | Attackers: TensorDict { <br>&emsp;**self**: $<power, presence>$ <br>&emsp;**others**: $<power, presence>$ <br>&emsp;**tasks**: $<state>$ <br> **batch_size**: $num\_envs$ } <br> Defenders: TensorDict { <br>&emsp;**self**: $<power, presence, location>$ <br>&emsp;**others**: $<power, presence, location>$ <br>&emsp;**tasks**: $<state>$<br> **batch_size**: $num\_envs$}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Observation Values | Attackers: <br>&emsp;<u>**self**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{attacker}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;<u>**others**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{attacker}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;<u>**tasks**</u><br>&emsp;&emsp;$state$: [$0$, $n_{network\_states}$] <br><br> Defenders: <br>&emsp;<u>**self**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{defender}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;&emsp;$location$: [$0$, $n_{subnetworks}$]<br>&emsp;<u>**others**</u><br>&emsp;&emsp;$power$: [$0$, $max\_power_{defender}$]<br>&emsp;&emsp;$presence$: [$0$, $1$]<br>&emsp;&emsp;$location$: [$0$, $n_{subnetworks}$]</u><br>&emsp;<u>**tasks**</u><br>&emsp;&emsp;$state$: [$0$, $n_{network\_states}$] |

---

## Usage

### Parallel API
```python
from free_range_zoo.envs import cybersecurity_v0

main_logger = logging.getLogger(__name__)

# Initialize and reset environment to initial state
env = cybersecurity_v0.parallel_env(render_mode="human")
observations, infos = env.reset()

# Initialize agents and give initial observations
agents = []

cumulative_rewards = {agent: 0 for agent in env.agents}

current_step = 0
while not torch.all(env.finished):
    agent_actions = {
        agent_name: torch.stack([agents[agent_name].act()])
        for agent_name in env.agents
    }  # Policy action determination here

    observations, rewards, terminations, truncations, infos = env.step(agent_actions)
    rewards = {agent_name: rewards[agent_name].item() for agent_name in env.agents}

    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name][0])  # Policy observation processing here
        cumulative_rewards[agent_name] += rewards[agent_name]

    main_logger.info(f"Step {current_step}: {rewards}")
    current_step += 1

env.close()
```

### AEC API
```python
from free_range_zoo.envs import cybersecurity_v0

main_logger = logging.getLogger(__name__)

# Initialize and reset environment to initial state
env = cybersecurity_v0.parallel_env(render_mode="human")
observations, infos = env.reset()

# Initialize agents and give initial observations
agents = []

cumulative_rewards = {agent: 0 for agent in env.agents}

current_step = 0
while not torch.all(env.finished):
    for agent in env.agent_iter():
        observations, rewards, terminations, truncations, infos = env.last()

        # Policy action determination here
        action = env.action_space(agent).sample()

        env.step(action)

    rewards = {agent: rewards[agent].item() for agent in env.agents}
    cumulative_rewards[agent] += rewards[agent]

    current_step += 1
    main_logger.info(f"Step {current_step}: {rewards}")

env.close()
```

---

## Configuration

```{eval-rst}
.. currentmodule:: free_range_zoo.envs.cybersecurity.env.structures.configuration

.. autoclass:: AttackerConfiguration
.. autoclass:: CybersecurityConfiguration
.. autoclass:: DefenderConfiguration
.. autoclass:: NetworkConfiguration
.. autoclass:: RewardConfiguration
.. autoclass:: StochasticConfiguration

```

<hr>

## API

```{eval-rst}
.. currentmodule:: free_range_zoo.envs.cybersecurity.env.cybersecurity

.. autoclass:: env
.. autoclass:: raw_env
```
