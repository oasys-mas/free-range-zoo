# Wildfire
```{toctree}
:caption: Environments
:hidden:

specification
baselines
```

## Description

The wildfire domain simulates a grid-based environment where fires can spread and agents are tasked with extinguishing
them by applying suppressant. The environment is dynamic and partially observable, with fires that can spread across
adjacent tiles and vary in intensity. Fires can also burn out once they reach a certain intensity threshold.

<u>**Environment Dynamics**</u><br>
- Fire Spread: Fires start at designated locations and spread to neighboring tiles, increasing in intensity over
  time. The intensity of the fire influences how much suppressant is needed to extinguish it. Fires will continue
  to spread until they either burn out or are controlled by agents.
- Fire Intensity and Burnout: As fires spread, their intensity increases, making them harder to fight. Once a
  fire reaches a critical intensity, it may burn out naturally, stopping its spread and extinguishing itself.
  However, this is unpredictable, and timely intervention is often necessary to prevent further damage.
- Suppression Mechanism: Agents apply suppressant to the fire to reduce its intensity. However, suppressant is a
  finite resource. When an agent runs out of suppressant, they must leave the environment to refill at a designated
  station before returning to continue fighting fires.

<u>**Environment Openness**</u><br>
- **agent openness**: Environments where agents can dynamically enter and leave, enabling dynamic cooperation and
  multi-agent scenarios with evolving participants.
    - `wildfire`: Agents can run out of suppressant and leave the environment, removing their contributions
      to existing fires. Agents must reason about their collaborators leaving, or new collaborators entering.
- **task openness**: Tasks can be introduced or removed from the environment, allowing for flexbile goal setting
  and adaptable planning models
    - `wildfire`: Fires can spread beyond their original starting point, requiring agents to reason about new
      tasks possibly entering the environment as well as a changing action space: Fires can spread beyond
      their original starting point, requiring agents to reason about new tasks possibly entering the
      environment as well as a changing action space.
- **frame / type openness**: Different frames (e.g. agent abilities or skills) can be added, removed, or modified,
  expending the environmental complexity and requiring agents to infer their neighbors changing abilities.
    - `wildfire`: Agents can damage their equipment over time, and have their capabilities slowly degrade. On
      the other hand, agents might also recieve different equipment upon leaving the environment to resupply.
