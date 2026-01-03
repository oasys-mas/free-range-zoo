# Rideshare
<p align="center">
    <img src="env/assets/rendering.gif" width="300" height="300">
</p>
## Description

The rideshare domain simulates a grid-based environment where apssengers can appear and agents are tasked with
delivering passengers from their current location to their desination. The environment is dynamic and partially
observable, where agents cannot observe the contents of another agents car.

<u>**Environment Dynamics**</u><br>
- Passenger Entry / Exit: Passengers enter the environment from outside the simulation at any space. They must be
  accepted by an agent and picked up at their current location, then dropped off at their destination. Agents
  recieve the fare defined by an individual task, and receive penalties if any passenger is waiting for a state
  transition for too long.

<u>**Environment Openness**</u><br>

- **task openness**: Tasks can be introduced or removed from the environment, allowing for flexbile goal setting and
  adaptable planning / RL models.
  - `rideshare`: New passengers can enter the environment, and old ones can leave. Agents have to reason about
    competition for tasks, as well as how to efficiently pool, overlap, and complete tasks.


# Specification

---

| Import             | `from free_range_zoo.envs import rideshare_v0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Actions            | Discrete & Deterministic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Observations       | Discrete and fully observed with private observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Parallel API       | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Manual Control     | No                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Agent Names        | [$driver$_0, ... , $driver$_n]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| # Agents           | $n$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Action Shape       | ($envs$, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Action Values      | [$[accept (0)\|pick (1)\|drop (2)]\_0$, ..., $[accept (0)\|pick (1)\|drop (2)]\_{tasks}$, $noop$ (-1)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Observation Shape  | TensorDict: { <br>&emsp;**self**: $<y, x, num_{accepted}, num_{riding}>$<br>&emsp;**others**: $<y, x, num_{accepted}, num_{riding}>$<br>&emsp;**tasks**: $<y, x, y_{dest}, x_{dest}, accepted_by, riding_by, entered_step>$ <br> **batch_size**: $num\_envs$ }                                                                                                                                                                                                                                                                                                                                                             |
| Observation Values | <u>**self**</u>:<br>&emsp;$y$:$[0, max_y]$<br>&emsp;$x$: $[0, max_x]$<br>&emsp;$num\_accepted$: $[0, pooling\_limit]$<br>&emsp;$num_riding$: $[0, pooling\_limit]$<br><u>**others**</u>:<br>&emsp;$y$:$[0, max_y]$<br>&emsp;$x$: $[0, max_x]$<br>&emsp;$num\_accepted$: $[0, pooling\_limit]$<br>&emsp;$num_riding$: $[0, pooling\_limit]$<u>**tasks**</u>:<br>&emsp;$y$: $[0, max_y]$<br>&emsp;$x$: $[0, max_x]$<br>&emsp;$y_{dest}$: $[0, max_y]$<br>&emsp;$x_{dest}$: $[0, max_x]$<br>&emsp;$riding\_by$: $[0, num_{agents}]$<br>&emsp;$accepted\_by$: $[0, num_{agents}]$<br>&emsp;$entered\_step$: $[0, max_{steps}]$ |

---
## Usage
### Parallel API
```python
from free_range_zoo.envs import rideshare_v0

main_logger = logging.getLogger(__name__)

# Initialize and reset environment to initial state
env = rideshare_v0.parallel_env(render_mode="human")
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
from free_range_zoo.envs import rideshare_v0

main_logger = logging.getLogger(__name__)

# Initialize and reset environment to initial state
env = rideshare_v0.parallel_env(render_mode="human")
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

## Rendering

For information on visualizing rideshare simulations, see the [Rendering Documentation](./env/utils/README.md). The rendering module provides tools to convert CSV logs into interactive or video-based visualizations using Pygame. See [rendering.py](./env/utils/rendering.py) for implementation details.

---
