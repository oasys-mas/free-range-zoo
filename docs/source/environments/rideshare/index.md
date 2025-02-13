# Rideshare
```{toctree}
:caption: Environments
:hidden:

specification
baselines
```

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

