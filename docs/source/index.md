---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:caption: Events
:hidden:
events/mosaei-2025/index
```

```{toctree}
:caption: Introduction
:hidden:
introduction/moasei
introduction/installation
introduction/basic_usage
```

```{toctree}
:caption: API
:hidden:
api/environments
api/logging
```

```{toctree}
:caption: Environments
:hidden:
environments/wildfire/index
environments/cybersecurity/index
environments/rideshare/index
```

```{toctree}
:caption: Development
:hidden:
Github <https://github.com/oasys-mas/free-range-zoo>
```

```{project-logo} _static/img/goat_main_logo.png
:alt: free-range-zoo Logo
```

This repository provides a collection of PyTorch implementations for various reinforcement learning / planning environments. It includes both [environments](https://github.com/oasys-mas/free_range_zoo/free_range_zoo/envs) and links 
[reasoning models](https://github.com/oasys-mas/free_range_zoo/models/) used for decision-making in these domains. The contained domains (`wildfire`, `rideshare`, `cybersecurity`), are designed with a special emphasis on **open-agent**, **open-task**, 
and **open-frame** systems. These systems are designed to allow dynamic changes in the environment, such as the entry and exit of agents, tasks, and types, simulating more realistic and flexible scenarios. 
The implementations are designed to facilitate experimentation, comparison, and the development of new techniques in RL and planning. 

 There are extensions for each environment to handle batching for updating the trajectories of multiple states simulataneously. Pettingzoo partially supports agent-open environments through its distinction between `agents` and `possible_agents`, but has no built-in support for task or frame openness.

# Core Research Applications
All forms of openness should have the ability to be completely removed the from the environment. Allowing for testing with each form in complete isolatioAll forms of openness should have the ability to be completely removed 
the from the environment. Allowing for testing with each form in complete isolation.

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

