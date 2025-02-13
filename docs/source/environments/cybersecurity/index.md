# Cybersecurity
```{toctree}
:caption: Environments
:hidden:

specification
baselines
```

## Description
The cybersecurity domain simulates a network environment where nodes are either attacked or patched by agents, with
the goal of protecting or exploiting the system. The environment is partially observable, with defenders needing to
respond to exploited nodes, while attackers aim to increase the exploited state of nodes. The dynamic interaction
between attackers and defenders creates an evolving cybersecurity landscape where agents must adapt to the changing
system state.

<u>**Dynamics**</u><br>
- Nodes: The network consists of multiple nodes, each of which can be in one of several states, ranging from
  unexploited to fully exploited. Exploited nodes represent compromised parts of the system that attackers have
  successfully infiltrated, while unexploited nodes are safe and intact.
- Exploited State: Nodes can be attacked by malicious agents to increase their exploited state, making them
  vulnerable to further exploitation. As nodes become more exploited, they pose a greater risk to the overall
  system.
- Patching and Exploiting: Nodes can be patched by defenders to reduce their exploited state, while attackers
  attempt to exploit unpatched or partially patched nodes to further their objectives. The environment is
  partially observable, meaning that defenders do not always know the state of all nodes, requiring them to
  take actions based on limited information.

<u>**Environment Openness**</u><br>
- **agent openness**: Environments where agents can dynamically enter and leave, enabling dynamic cooperation and
  multi-agent scenarios with evolving participants.
    - `cybersecurity`: Agents can lose access to the network, disallowing them from taking actions within the
      environment for a period of time. Agents must reason about how many collaborators are within the
      environment with them, and whether they are able to sufficiently fight opposing agents.
