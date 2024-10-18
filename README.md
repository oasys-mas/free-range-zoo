# free-range-zoo

This repository provides a collection of PyTorch implementations for various reinforcement learning / planning environments. It includes both [environments](free_range_zoo/free_range_zoo/envs) and links 
[reasoning models](models/) used for decision-making in these domains. The contained domains (`wildfire`, `rideshare`, `cybersecurity`), are designed with a special emphasis on **open-agent**, **open-task**, 
and **open-frame** systems. These systems are designed to allow dynamic changes in the environment, such as the entry and exit of agents, tasks, and types, simulating more realistic and flexible scenarios. 
The implementations are designed to facilitate experimentation, comparison, and the development of new techniques in RL and planning. 

The existing environments in this repository all utilize the same AEC environment API as the conventional MARL environments provided by [pettingzoo](https://github.com/Farama-Foundation/PettingZoo). There are extensions for
each environment to handle batching for updating the trajectories of multiples states simulataneously. Pettingzoo partially supports agent-open environments through its distinction between `agents` and `possible_agents`, but 
has no built-in support for task or frame openness.

### Core Research Applications
All forms of openness should have the ability to be completely removed the from the environment. Allowing for testing with each form in complete isolatioAll forms of openness should have the ability to be completely removed 
the from the environment. Allowing for testing with each form in complete isolation.

- **agent openness**: Environments where agents can dynamically enter and leave, enabling ad-hoc teamwork and multi-agent scenarios with evolving participants.
    - Environments:
        - `wildfire`: Agents can run out of suppressant and leave the environment, removing their contributions to existing fires. Agents must reason about their collaborators leaving, or new collaborators entering.
        - `cybersecurity`: Agents can lose access to the network, disallowing them from taking actions within the environment for a period of time. Agents must reason about how many collaborators are within the environment
                           with them, and whether they are able to sufficiently fight opposing agents.
- **task openness**: Tasks can be introduced or removed from the environment, allowing for flexbile goal setting and adaptable planning models
    - Environments:
        - `wildfire`: Fires can spread beyond their original starting point, requiring agents to reason about new tasks possibly entering the environment as well as a changing action space: Fires can spread beyond 
                      their original starting point, requiring agents to reason about new tasks possibly entering the environment as well as a changing action space.
        - `rideshare`: New passengers can enter the environment, and old ones can leave. Agents have to reason about competition for tasks, as well as how to efficiently pool, overlap, and complete tasks.
- **frame / type openness**: Different frames (e.g. agent abilities or skills) can be added, removed, or modified, expending the environmental complexity and requiring agents to infer their neighbors changing abilities.
    - Environments:
        - `wildfire`: Agents can damage their equipment over time, and have their capabilities slowly degrade. On the other hand, agents might also recieve different equipment upon leaving the environment to resupply.

## Documentation

A description of repository and domain structures are given below. For more comprehensive documentation visit our [documentation]() page.

### Domain structure

The structure of each domain definition is described below and is mostly consistent across domains:

```python
envs
├── <environment>               #   <Environment implementation>
│   ├── configs                 #       Benchmark configurations
│   └── env                     #       Environment definitions
│       ├── spaces              #           Action / observation spaces
│       ├── structures          #           Configuration settings and state
│       ├── transitions         #           Environment transition functions
│       ├── utils               #           Misc. calculation / generation utilities
│       └── <environment>.py    #       Main environment definition
└── <environment>_vX.py         # Environment import file
```

### Repository Structure

The structure of the repository described below:

```python
free_range_zoo
├── experiments                         # Experimental code
├── free_range_zoo
│   ├── envs                            # Environment implementations
│   │   ├── cybersecurity               #   Cybersecurity
│   │   ├── rideshare                   #   Rideshare
│   │   └── wildfire                    #   Wildfire
│   ├── utils                           # Converters / environment abstract classes
│   └── wrappers                        # Model wrappers and utilities
├── models                              # Model code (planning / MOHITO / ddqn)
├── notebooks                           # Environment runtime notebooks (primarily for development)
├── tests                               # Tests
│   ├── free_range_zoo
│   │   ├── envs                        #   Tests for all environment utilities
│   │   └── utils                       #   Tests for all package utilities
│   ├── profiles                        # Environment performance profiles
│   └── utils                           # Testing utilities
├── README.md
├── poetry.lock
└── pyproject.toml                      # Package dependencies and package definition
```

### Interacting with environments
Interaction with environments is a relatively simple process. All environments follow the policy of `observe -> act -> environment_step -> repeat` consistent with
the standard agent environment cycle (AEC). The state of the environment will be updated once all actions have been submitted, and will return as below. An example
of interacting with the AEC cycle is shown below.

```python
env = <environment>.parallel_env(<environment arguments>)
observations, infos = env.reset()

# Initialize agents and give initial observations

while not torch.all(env.finished):
    agent_actions = {agent_name: torch.stack([agents[agent_name].act()]) for agent_name in env.agents} # Policy action determination here

    observations, rewards, terminations, truncations, infos = env.step(agent_actions)
    rewards = {agent_name: rewards[agent_name].item() for agent_name in env.agents}

    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name][0]) # Policy observation processing here
        cum_rewards[agent_name] += rewards[agent_name]

    main_logger.info(f"Step {current_step}: {rewards}")
```

## Roadmap

[TODO.md](TODO.md)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Repository Authors

- [Ceferino Patino](https://www.github.com/C4theBomb)[^1]
- [Daniel Redder](https://github.com/daniel-redder)[^1]

[^1]: Primary Maintainers

## Published Works

Each paper's code is stored as a release. We maintain old configurations for experiments for these works in [experiments](experiments/) for comparison purposes, however
with development changes we cannot guarantee complete backwards compatibility. For complete repoducibility, see each paper's respective release

- [AAAI-2025](https://github.com/oasys-mas/free-range-zoo/releases/tag/AAAI-2025-1.0) - Note the parameters listed and parameters describe in our technical appendix
    - [training scripts](experiments/mohito/)
    - [plotting](notebooks/plotting/plotting_readme.md)

## Used By

This project has been developed and utilized in collaboration by the following organizations:

- University of Georgia - Athens -  Athens, GA
- University of Nebraska - Lincoln -  Lincoln, NE
- Oberlin College - Oberlin, OH
