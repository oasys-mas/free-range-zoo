# Environment API
Our environments differ slightly from `pettingzoo` in two aspects. Our environments operate in vectorized **batches**. 
The intention of this is to take advantage of batched computation provided by **SIMD** architectures, and allow for 
multithreaded operations to make environment transitions faster. Additionally, our environments use standard 
`Configuration` objects to organize environment parameters.

---

## Directory Structure

Each environment follows the same directory structure for defining environment dynamics.

```
envs
├── <environment>               #   <Environment implementation>
│   ├── configs                 #       Benchmark configurations
│   └── env                     #       Environment definitions
│       ├── spaces              #           Action / observation spaces
│       ├── structures          #
|       | |_ configuration.py   #           "Configuration" for env settings
|       | |_ state.py           #           "State" class to store curr state
│       ├── transitions         #           Environment transition functions
│       ├── utils               #           Misc. tools
│       └── <environment>.py    #       Main environment definition
└── <environment>_vX.py         # Environment import file
```

---

## Importing Environments
The method for importing environments for utilization is identical to `pettingzoo`.

``` python
# free-range-zoo
from free_range_zoo.envs import wildfire_v0

# vs. pettingzoo
from pettingzoo.butterfly import pistonball_v6
```

The environment can then be used in a manner similar to `pettingzoo` environements.

---

## Configuring Environments
In order to make environments more flexible and parameterize transition functions, `free-range-zoo` utilizes 
`Configuration` classes to group parameters for the environment. Each environment's respective 
`Configuration` objects and structure are documented in their **Specification** page, and implementations 
are included in `envs/<environment_name>/env/structures/configuration.py`. 

<!--TODO: Add in the link for generator documentation-->
For simplicity, generators for configurations have been included with the competition materials. Configuration
generators for each environment are documented [here]().

Each environment module exposes two methods `parallel_env()` and `aec_env()`. `parallel_env` is more commonly used and
follows the Parallel AEC loop where all agent actions are submitted simultaneously.

Both functions take in several parameters:
- `configuration`: `Configuration` - the configuration object for the environment to use at initialization.
- `parallel_envs`: `int` - the number of parallel episodes to run simultaneously within the environment.
- `max_steps`: `int` - the number of environment steps to run before termination. `wildfire` has extra termination 
  criteria.
- `device`: `torch.DeviceObjType` - the device to use for environment data storage and processing. All outputs from 
  the environment will also be on this device.
- `single_seeding`: `bool` - Whether all environment parameters should operate on a single seed. This parallelizes 
  random generation across environments rather than having each environment have its own seed. This significantly
  increases speed for each environment step.
- `buffer_size`: `int` - The size of the buffer for random generation. Generally this should be set to the value of
  `max_steps * parallel_envs` to make it so that random generation is only done once per complete episode.

There are additional parameters available in each environment file. Note that action space and observability parameters
will be kept on the default values during evaluation, so it is advised that the same is done during training so that
observations are kept consistent with evaluation.

<!--TODO: Add in the link for the environment usage documentation-->
To see more details about how to interact with environments, see [Environment Usage]().

---

## Actions

Actions in `free-range-zoo` have a consistent structure, however must be represented in a way that is compatible with
**task openness**. Actions have two categories, **task actions**, which are represented by the action input 
`(task id, action id)`, where action id is subject to change based on what actions are available to the agent. The 
other category is **task agnostic actions**, which are represented by action input `(padding, action id)`. Action ids 
for task agnostic actions are always consistent and always negative.

The reason why we represent action spaces in this way is due to having an unbounded number of tasks due to **task
openness**. In order to handle this, action spaces are non-static and must grow and shrink during execution in order
to accomodate additional tasks.

**Task agnostic actions** represent actions for which the association with an individual task is not known. An example
of this would be `noop` in all domains. `noop` has an effect within the environment, however that effect is independent
from any individual task. **Task actions** are actions with which a direct relationship can be drawn between the task
and the associated action. For example, `pick` within rideshare has a clear association with an inidividual task, and 
thus must be targeted towards a specific task.

---

## Additional Useful Attributes

### Methods

`reset(seed: List[int], *args, **kwargs)`: Used for resetting the environment state to the initial state defined on
initialization by `configuration`. Takes initial seeds for each environment.

`step(actions: torch.Tensor, *args, **kwargs)`: Used to advance the entire environment a single step. Actions must be
provided in the form of `IntTensor` with shape `(parallel_envs, 2)`, where element (1) of the actions represents the
task which the agent is acting on, and (2) represents the action taken on that task.

`action_space(agent: str)`: Used to retrieve the action space of `agent`. Will return spaces which are similar to 
`gymnasium.Space` objects, but provide faster functionality.

`observation_space(agent: str)`: Used to retrieve the observation space of `agent`. Will return spaces which are similar to 
`gymnasium.Space` objects, but provide faster functionality.

`state()`: Used to retrive the current state of the environment. `State` will be the vectorized state of all parallel
environments at the time of execution.

### Attributes

`agents`: The list of all agents names in all environments.

`agent_name_mapping`: The mapping of agent names to agent indices for all environments.

`environment_task_count`: An `IntTensor` representing the number of tasks currently present in each parallel environment.
The tensor has the shape `(parallel_envs)`.

`agent_task_count`: An `IntTensor` reprsenting the number of tasks available to each agent. Holds the form 
`Dict[str, IntTensor]` and is keyed by agent name, where `IntTensor` has shape `(parallel_envs)`.

`task_store`: A ragged tensor of the task observations for all environments. Individualized per environment.

`agent_action_mapping`: A mapping of task indices from an agent's action space to `task_store`. This is used to map actions
from an individual agent's action space back to the global task store. Holds the form of `Dict[str, IntTensor]` and is keyed
by agent name, where `IntTensor` is ragged, of shape `(parallel_envs, tasks)` and is representative of indices in 
`task_store`.

`agent_observation_mapping`: A mapping of task indices from an agent's observation space to `task_store`. This is used to 
map observations from an individual agent's observation space back to the global task store. Holds the form of 
`Dict[str, IntTensor]` and is keyed by agent name, where `IntTensor` is ragged, of shape `(parallel_envs, tasks)` and 
is representative of indices in `task_store`.

`terminated`: A dict of termination state of every current agent in every environment at the time called. The returned
dict is keyed by agent name and returns a `BoolTensor` of the termination state of the agent in each parallel 
environment. The inner tensors have the shape `(parallel_envs)`.

`truncated`: A dict of trunctation state of every current agent in every environment at the time called. The returned
dict is keyed by agent name and returns a `BoolTensor` of the truncation state of the agent in each parallel 
environment. The inner tensors have the shape `(parallel_envs)`.

`finished`: A dict of `terminated | truncated`. Holds the same form as `terminated` and `truncated`. The inner tensors have the shape `(parallel_envs)`.

