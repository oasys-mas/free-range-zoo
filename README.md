# free-range-zoo

This package introduces three new components which extend the MARL environment package [pettingzoo](https://github.com/Farama-Foundation/PettingZoo)  to open environments. Pettingzoo already supports agent open environments through its distinction between `possible_agents` and `agents`, but this does not account for [Task Open Environments]() and [Agent Type Openness](). Thus this work includes a extension of the pettingzoo `AECEnv` which supports all the existing gymnasium/supersuit/pettingzoo wrappers, but also introduces new components helpful for these kinds of openness. 

This package includes three pieces
- Example Open Environments: Dynamic-Wildfire, and Dynamic-Rideshare
- Observation Wrappers: wrappers which convert a pettingzoo style observation to `incidence-graphs` of various forms as seen in [paper]()
- Modified AECEnv (TODO): a modified version of AECEnv which introduces tracking agent types, the use of a generator for `possible_agents` instead of a bounded agent set, and a modified AECEnv interaction loop which can include the agent's type

## Environments



#Wrappers



##DynamicAECEnv
