![Free-Range-Zoo logo, a goat standing on a hill.](https://github.com/oasys-mas/free-range-zoo/blob/main/docs/source/_static/img/goat_main_logo.png)

# free-range-zoo

This repository provides a collection of PyTorch implementations for various reinforcement learning / planning environments. It includes [environments](free_range_zoo/free_range_zoo/envs) used for decision-making in these domains. The contained domains (`wildfire`, `rideshare`, `cybersecurity`), are designed with a special emphasis on **open-agent**, **open-task**, 
and **open-frame** systems. These systems are designed to allow dynamic changes in the environment, such as the entry and exit of agents, tasks, and types, simulating more realistic and flexible scenarios. Such environments are illustrated in this [AI magazine article](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12131). This api is designed to facilitate experimentation, comparison, and the development of new techniques in RL and planning for open environments. 

The existing environments in this repository all utilize the same AEC/Parallel environment API as the conventional MARL environments provided by [pettingzoo](https://github.com/Farama-Foundation/PettingZoo). However, we provide three main features convenient for openness: 
- **(1) Batched training**, mulitple independent environments can be executed in parallel via vectorized operations.
- **(2) Fast spaces**, courtesy of [free-range-rust](https://github.com/c4patino/free-range-rust) we use `rust` based vectorized spaces. These significantly increase the speed of representing changing spaces in openness.
- **(3) Easy GPU execution**, our environments can be used on gpu by simply providing a `torch.Device` on environment construction.
> [!note]
> Due to the nature of RL domains, a large number of small sequential calculations, CPU execution is faster except when using a large number of parallel environments, as possible with (1). 

### Core Research Applications
All forms of openness can be enabled or disabled in each environment through settings defined in `<environment>/env/structures/configuration.py`

- **agent openness**: Environments where agents can dynamically enter and leave, enabling dynamic cooperation and multi-agent scenarios with evolving participants.
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

A description of this repository structure is given below. **For a quick start guide and documentation visit our [documentation](https://oasys-mas.github.io/free-range-zoo/) page.**

### Domain structure

The structure of each domain definition is described below and is mostly consistent across domains:

```
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

```
free_range_zoo
├── experiments                         # put your experiments here.
├── free_range_zoo
│   ├── envs                            # Environment implementations
│   │   ├── cybersecurity               #   Cybersecurity
│   │   ├── rideshare                   #   Rideshare
│   │   └── wildfire                    #   Wildfire
│   ├── utils                           # Converters / environment abstract classes
│   └── wrappers                        # Model wrappers and utilities
├── models                              # Model code (planning / MOHITO / ddqn)
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

## Roadmap

[TODO.md](TODO.md)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Repository Authors

- [Ceferino Patino](https://www.github.com/C4theBomb)[^1]
- [Daniel Redder](https://github.com/daniel-redder)[^1]
- [Alireza Saleh Abadi](https://github.com/bboyfury)[^1]

[^1]: Primary Maintainers


## Used By

This project has been developed and utilized in collaboration by the following organizations:

- University of Georgia - Athens -  Athens, GA
- University of Nebraska - Lincoln -  Lincoln, NE
- Oberlin College - Oberlin, OH
