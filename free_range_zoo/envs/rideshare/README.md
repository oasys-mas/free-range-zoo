# Rideshare
## Description

# Specification

---

| Import             | `from freerangezoo.otask import rideshare_v0` |
|--------------------|------------------------------------|
| Actions            | Discrete and perfect                            |
| Observations | Discrete and fully observed with private observations |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agent Names             | [$driver$_0, ..., $driver$_n] |
| #Agents             |    $n$                                  |
| Action Shape       | (envs, 2)                 |
| Action Values      | [-1, $\|tasks\|$], [-1,2]\*                    |
| Observation Shape | TensorDict: { <br> &emsp; **Agent's self obs**, <ins>'self'</ins>: 5 `<agent index, ypos, xpos, #accepted passengers, #riding passengers>`, <br> &emsp; **Other agent obs**, <ins>'others'</ins>: ($\|Ag\| \times 5$) `<agent index, ypos, xpos, #accepted passengers, #riding passengers>`, <br> &emsp; **Fire/Task obs**, <ins>'tasks'</ins>: ($\|X\| \times 5$) `<task index, ystart, xstart, yend, xend, acceptedBy, ridingWith, fare, time entered>` <br> **batch_size: `num_envs`** <br>}|
| Observation Values   | <ins>self</ins> <br> **agent index**: [0,n), <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **number accepted passengers**: [0, $\infty$), <br> **number riding passengers**: [0,$\infty$) <br> <br> <ins>others</ins> <br> **agent index**: [0,$n$), <br> **ypos**: [0,grid_height], <br> **xpos**: [0, grid_width], <br> **number accepted passengers**: [0, $\infty$), <br> **number riding passengers**: [0,$\infty$)  <br> <br> <ins>tasks</ins> <br> **task index**: [0, $\infty$), <br> **ystart**: [0,grid_height], <br> **xstart**: [0, grid_width], <br> **yend**: [0, grid_height], <br> **xend**: [0,grid_width] <br> **accepted by**: [0,$n$) <br> **riding with**: [0,$n$), <br> **fare**: (0, $\infty$), <br> **time entered**: [0,max steps] |

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

---
