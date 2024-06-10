# free-range-zoo

This package introduces three new components which extend the MARL environment package [pettingzoo](https://github.com/Farama-Foundation/PettingZoo)  to open environments. Pettingzoo already supports agent open environments through its distinction between `possible_agents` and `agents`, but this does not account for [Task Open Environments]() and [Agent Type Openness](). Thus this work includes a extension of the pettingzoo `AECEnv` which supports all the existing gymnasium/supersuit/pettingzoo wrappers, but also introduces new components helpful for these kinds of openness. 

----
**TODOS**

 > Base
- [ ] Implement possible agents as a generator

- [ ] Track agent types via OAECEnv

- [x] Implement the Incidence Graph Wrapper (working on action reshaping)

- [x] Confirm compatibility with `oaec_to_parallel_wrapper` (a slightly modified `aec_to_parallel_wrapper` from pettingzoo)

- [ ] Make pettingzoo style web-docs

> Environments
- [x] Implement the Dynamic-Wildfire Environment
- [x] Update Wildfire Documentation
- [ ] Migrate Wildfire render() script

=======

- [ ] Implement the Dynamic-Rideshare Environment
- [ ] Implement a Rideshare render()





## This package includes three pieces
- **Example Open Environments**: Dynamic-Wildfire, and Dynamic-Rideshare
- **Observation Wrappers**: wrappers which convert a pettingzoo style observation to `incidence-graphs` of various forms as seen in [paper]()
- **Modified OAECEnv**: a modified version of AECEnv which introduces tracking agent types, the use of a generator for `possible_agents` instead of a bounded agent set.

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

Thus each environment is implemented to run on any pytorch device. All environment functions are already wrapped in `torch.no_grad()'s` to prevent unintentional gradient propogation. We vectorize most of the operations in these environments which opens the possibility to have "batches" of independent environments training. As averaging over multiple training runs is common in evaluation, you can use `parallel_envs` to choose how many of these environments to use (this has nothing to do with parallel vs AEC APIs that is addressed later). This results in a few changes to the typical gymnasium structure:

> Pettingzoo Components -> free-range-zoo components
```py 
env.observations: Dict[ agent, List[]] -> observations: List[  Dict[agent, pytorch tensor ] per parallel_env]

env.terminations: Dict[ agent,  bool ] -> Dict[ AllList* [ bool ] ]
truncations ^

env.rewards: Dict[ agent, float ] -> Dict[ agent, List[ float ] per parallel_env]

#expected changes to the returned spaces
action_space : Dict[ agent, Gymnasium.Space] -> List[ Dict[ agent, Gymnasium.Space] per parallel_env]
observation_space ^
```
*[AllList]() is a typical python list, but when evaluated like `if thisAllList` it is treated as true if all elements in it are true.



### Creating a environment

To create a open task (OT) environment Use the [OACEnv]() class, follow the above changes for batching. The following changes are also neccessary for the wrapper to work:

> **self.agent_action_to_task_mapping**: `Dict`[ agent, tensor[ `parallel_envs`, `hyperedge_task_action_foreign_keys` ]  ]
> > A mapping of agent--> foreign keys for that agent which are available for that agent to take action on (if it is in here then it gets a hyperedge)
> 
> **self.tasks_in_environment**: `List`[ tensor[`hyperedge_task_action_foreign_keys` ] of `parallel_envs` length]
> > A ordered list of all tasks in the environment. This is used to establish order for nodes and hyperedges with the wrapper. This is agent independent. 



# Wrappers

<span style="color:red">
Warning: This wrapper only supports u-nary actions atm, what do I mean: only one ACTION can be associated with a task on a specific time step. This should not effect wildfire or rideshare. 
</span>



## [IncidenceGraph]()

is a wrapper made for OT environments. This wrapper converts our generic OT observation shape into a `torch_geometric` observation graph, and a "critic" graph which contains all agent observed information. This graph is comprised of `agent`, `task`, `action`, and `hyperedges`. The `hyperedges` represent a unique **possible** pairing of `task<-->action<-->agent`.

To use this wrapper you need to create or use a prexisting wrapper instance for a environment. This instance specifies:

- Which part of the task observation goes to task vs action nodes; `task_observation_components`, and respectively`action_observation_components`. 

- The index of a key (in the agent observation) which uniquely identifies agents; `agent_observation_identifier`

- A "foreign key" which matches task<-->action, think task position or rider id, **must be unique to a task<-->action pair**; `hyperedge_task_action_foreign_keys`

- The largest of the length of the observation, action, task observation, and other agent observation (does not include the "NODE_TYPE"); `feature_size`

- Which directionality of connecting nodes should be used, ("undirected", and "thesis") are implemented, either nodes have undirected edges, or edges facing hyperedges respectively; `incidence_graph_style`

- is a noop_action present (assume there is only one action which follows the "thesis" approach to task nodes) T/F; `noop_action` 
  
> For example the wildfire wrapper is:

```py
def wildfire_IncidenceGraph_v1_wrapper(env, incidence_graph_style="undirected", noop_action="one_task"):
    """
    Basic configuration details for the wildfire IncidenceGraph_v1 wrapper.
    """
    return IncidenceGraph_v1(
        env=env,
        
        #mapping task information to task/action nodes
        action_observation_components=[0,1], #fire position x,y
        task_observation_components=[0,1,2], #fire position x,y, fire intensity

        agent_observation_identifier=[0,1], #agent position x,y
        
        #mapping agent --> task --> action
        hyperedge_task_action_foreign_keys=[0,1], #fire position x,y  
        #!(index of this list is the index of the list given by agent_action_to_task_mapping)
    
        feature_size=4, #does not include the node type identifier

        incidence_graph_style=incidence_graph_style,

        noop_action=noop_action
    )
```

## oaec-to-parallel-wrapper

Just the `oaec-to-parallel-wrapper` from pettingzoo, but with a modified reward structure to account for batching. 