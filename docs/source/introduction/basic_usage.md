# Basic Usage

---

## Overview
This document explains:
1. The **interface** agents must follow.
2. The **environment run loop**, with a detailed breakdown of its components.
3. A step-by-step walkthrough of the `StrongestBaseline` agent for the wildfire domain, which utilizes a greedy policy.

---

## 1. **Agent Interface**
Agents interacting with the environment must follow a standard interface defined by the `Agent` abstract class, see [agent.py](https://github.com/oasys-mas/free-range-zoo/blob/main/free_range_zoo/utils/agent.py). Here's a breakdown of its structure:

### Initialization
```python
def __init__(self, agent_name: str, parallel_envs: int) -> None:
    """
    Args:
        agent_name: str - Name of the agent (used for identification).
        parallel_envs: int - Number of environments being run in parallel.
    """
```
- **`agent_name`**: A unique identifier for the agent.
- **`parallel_envs`**: Specifies how many environments the agent operates in simultaneously.

### Action Selection (`act`)
```python
def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
    """
    Args:
        action_space: free_range_rust.Space - Current action space available to the agent.
    Returns:
        List[List[int]] - List of actions, one for each parallel environment.
    """
```
- **Purpose**: Determines what actions the agent takes given the current state of the action space.
- **Input**: `action_space`, which defines the legal actions at a given time.
- **Output**: A list of actions (one per parallel environment).

### Observation Handling (`observe`)
```python
def observe(self, observation: Dict[str, Any]) -> None:
    """
    Args:
        observation: Dict[str, Any] - Observation from the environment.
    """
```
- **Purpose**: Processes the agent's observations from the environment.
- **Input**: `observation`, a dictionary containing relevant state information.

---

## 2. **Environment Run Loop**

The environment loop is the core process that executes interactions between agents and the environment. Here's the implementation and a step-by-step breakdown:

### Implementation
```python
from free_range_zoo.envs import cybersecurity_v0
import logging

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

### Step-by-Step Breakdown

#### Initialization
1. **Environment Setup**:
   ```python
   env = cybersecurity_v0.parallel_env(render_mode="human")
   observations, infos = env.reset()
   ```
   - Creates the environment with a parallel structure.
   - **Reset** initializes the environment to its initial state and provides:
     - `observations`: Initial observations for all agents.
     - `infos`: Metadata (e.g., environment-specific details).

2. **Agent Initialization**:
   ```python
   agents = []
   cumulative_rewards = {agent: 0 for agent in env.agents}
   ```
   - The `agents` dictionary maps agent names to their corresponding agent instances.
   - `cumulative_rewards` tracks the total rewards accumulated by each agent.

#### Interaction Loop
3. **Agent Actions**:
   ```python
   agent_actions = {
       agent_name: torch.stack([agents[agent_name].act()])
       for agent_name in env.agents
   }
   ```
   - Each agent determines an action using its policy.
   - Actions are grouped into `agent_actions`, which is then passed to the environment.

4. **Environment Step**:
   ```python
   observations, rewards, terminations, truncations, infos = env.step(agent_actions)
   ```
   - Executes the actions in the environment.
   - Returns:
     - `observations`: Updated state information.
     - `rewards`: Rewards for each agent.
     - `terminations` and `truncations`: Flags indicating the end of an episode.
     - `infos`: Additional metadata.

5. **Observation and Reward Handling**:
   ```python
   for agent_name, agent in agents.items():
       agent.observe(observations[agent_name][0])
       cumulative_rewards[agent_name] += rewards[agent_name]
   ```
   - Agents process new observations.
   - Accumulate rewards for each agent.

6. **Logging**:
   ```python
   main_logger.info(f"Step {current_step}: {rewards}")
   ```
   - Logs the rewards at each step for debugging or analysis.

#### Termination
7. **Close Environment**:
   ```python
   env.close()
   ```
   - Cleans up the environment resources.

---

## 3. **Sample Agent Implementation**

Below is an exmaple of an agent implementation. This is an implementation of a simple heuristical baseline for the
wildfire domain. Agents for RL solutions will likely have a different implementation for observation and actor logic,
and the method of implementation for more complicated solutions is left to the user.

### Implementation

```python
class StrongestBaseline(Agent):
    """Agent that always fights the strongest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        has_suppressant = self.observation['self'][:, 3] != 0
        fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, 3]

        argmax_store = torch.empty_like(self.t_mapping)

        for batch in range(self.parallel_envs):
            for element in range(self.t_mapping[batch].size(0)):
                argmax_store[batch][element] = fires[batch][element]

            if len(argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # There are no fires in the environment so agents have to noop
                continue

            self.actions[batch, 0] = argmax_store[batch].argmax(dim=0)
            self.actions[batch, 1] = 0

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
```

### Step-by-Step Breakdown

1. **Initialization**

The agent is initialized with the number of parallel environments (parallel_envs) and its name (agent_name). A tensor 
`self.actions` is created to store the agent's actions for all parallel environments. It has a shape of 
`(parallel_envs, 2)` with an initial value of zeros.

```python
...
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)
...
```

2. **Acting (`act` method)**

The act method simply returns the current action tensor (self.actions). The actions are updated in the observe method 
based on the observations provided by the environment.

```python
...
def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
    return self.actions
...
```

3. **Observing (`observe` method)**
    1. Inputs: The agent receives observation data from the environment. This contains: 
        - `self.observation`: The agent's observation of environment's current state.
        - `t_mapping`: A mapping for agent actions in the environment.

        ```python
        ...
        def observe(self, observation: Dict[str, Any]) -> None:
        ...
        ```

    2. **Extracting Observations**: self.t_mapping is extracted from observation under the key 'agent_action_mapping'.
       A tensor of fire strengths is extracted from the observation's tasks. This is padded with `-100` to handle 
       variable-length tasks across parallel environments. `has_suppressant` is a boolean tensor indicating whether each 
       agent has suppressant available. This is derived from the `self` key in the observation. Read more about the
       structure of the structure of each environment's observation space in its **environment page**.

       ```python
       ...
       self.observation, self.t_mapping = observation
       self.t_mapping = self.t_mapping['agent_action_mapping']

       has_suppressant = self.observation['self'][:, 3] != 0
       fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, 3]
       ...
       ```
    3. **Preparing to Identify Strongest Fires**: A temporary tensor, argmax_store, is created to store fire strengths 
       mapped to t_mapping for each batch (parallel environment).
       ```python
       ...
       argmax_store = torch.empty_like(self.t_mapping)
       ...
       ```

    4. **Iterating Through Parallel Environments**: For each batch (parallel environment) fire strengths are mapped 
       into argmax_store using t_mapping. If no fires are available in the environment, the agent sets its actions 
       to [-1, -1], indicating a "no-op" (no action). The index of the strongest fire in the batch is found using 
       argmax. The action's first element (actions[batch, 0]) is set to this index, and the second element 
       (actions[batch, 1]) is set to 0.
       ```python
       ...
       for batch in range(self.parallel_envs):
           for element in range(self.t_mapping[batch].size(0)):
               argmax_store[batch][element] = fires[batch][element]

           if len(argmax_store[batch]) == 0:
               self.actions[batch].fill_(-1)  # There are no fires in the environment so agents have to noop
               continue

           self.actions[batch, 0] = argmax_store[batch].argmax(dim=0)
           self.actions[batch, 1] = 0
       ...
       ```

    5. **Handling Agents Without Suppressant**: The second element of the action (actions[:, 1]) is set to -1 (no-op) 
       for agents without suppressant using masked_fill.
       ```python
       ...
       self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop
       ...
       ```
