# free-range-zoo

This package introduces three new components which extend the MARL environment package [pettingzoo](https://github.com/Farama-Foundation/PettingZoo)  to open environments. Pettingzoo already supports agent open environments through its distinction between `possible_agents` and `agents`, but this does not account for [Task Open Environments]() and [Agent Type Openness](). Thus this work includes a extension of the pettingzoo `AECEnv` which supports all the existing gymnasium/supersuit/pettingzoo wrappers, but also introduces new components helpful for these kinds of openness. 

----
TODOS

- [x] Implement the Dynamic-Wildfire Environment
- [ ] Implement the Incidence Graph Wrapper (working on action reshaping)
- [ ] Implement possible agents as a generator
- [x] Update Wildfire Documentation
- [ ] Migrate Wildfire render() script
- [ ] finish base README


----



This package includes three pieces
- **Example Open Environments**: Dynamic-Wildfire, and Dynamic-Rideshare
- **Observation Wrappers**: wrappers which convert a pettingzoo style observation to `incidence-graphs` of various forms as seen in [paper]()
- **Modified OAECEnv**: a modified version of AECEnv which introduces tracking agent types, the use of a generator for `possible_agents` instead of a bounded agent set, and a modified AECEnv interaction loop which can include the agent's type along with the agent whose turn it is

## Environments

Our goal with these open environments is to make them **reproducible**, **fast**, and **readable**. Towards that end configuration settings for experiments we use this framework for can be imported via the `paper_configurations` object associated with each environment. 

To minimize time spent waiting for i/o between a gpu loaded model and a cpu executed environment, we implement all of these environments in torch. 

We provide documentation with each environment along with thorough inline-comments. 





> This is a example environment initialization for dynamic-wildfire
```py 
from envs.otask.wildfire import wildfire_v0, paper_configurations

grid_configuration, stoch_configuration = paper_configurations[0]()

env = wildfire_v0(
    parallel_envs=1, #selects the "batch" size of the environment
    device = 'cuda:0' #device to run the environment on

    #configuration @dataclasses / Dictionaries
    grid_conf=grid_configuration, 
    stochastic_conf=stoch_configuration,  
)
```







### Batching

Thus each environment is implemented to run on any pytorch device. All environment functions are already wrapped in `torch.no_grad()'s` to prevent unintentional gradient propogation. 

We vectorize most of the operations in these environments which opens the possibility to have "batches" of independent environments training. As averaging over multiple training runs is common in evaluation, you can use `parallel_envs` to choose how many of these environments to use. 


When accounting for openness as well that gives us a somewhat complex observation.




### Wildfire

A task-open environment based off of [cite wildfire source](). 


# Wrappers



## DynamicAECEnv
