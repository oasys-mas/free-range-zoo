# Wildfire

<!-- Observations are always implemented with a **Tuple** space containing: a single `np.int` **Box** space for task-global observations, and multiple `np.int` **Box** spaces for task observations. **The exception** to this is if one of the observations is not a integer, in which case the observations spaces listed here are what is actually implemented. <ins>**This environment uses Box**</ins>. We show the observation spaces not as **Box**es because it is easier to interprete that way. Please look at the `observation_space` function in each environment before using it.  -->

----

| Import             | `from free_range_zoo.envs import wildfire_v0` |
|--------------------|------------------------------------|
| Actions            | Discrete \& Stochastic                            |
| Observations | Discrete \& Fully Observed with private observations [^1]
| Parallel API       | Yes                                |
| Manual Control     | No                                 
|
| Agent Names             | ['$`firefighter\_0`$', ..., '$`firefighter\_n`$']` |
| #Agents             |    $`n`$                                  |
| 
Action Shape       | (envs, 2)              |
| Action Values      |  [-1, '$`\|X\|`$'], \[0\][^2]              
|
| Observation Shape | TensorDict: { <br> &emsp; <bf>Agent's self obs, 'self'</bf>: 4 `<y, x, fire power, suppressant\>`, <br> &emsp; <bf>Other agent obs, 'others'</bf>: ('$`\|Ag\| \times 4`$') <y,x,fire power, suppressant\>, <br> &emsp; <bf>Fire/Task obs, 'fire'</bf>: ('$`\|X\| \times 4`$') <y, x, fire level, intensity> <br> <bf>batch\_size: `num\_envs`</bf> <br>}|
| Observation Values   | <ins>Self</ins> <br> <bf>y</bf>: [0,grid\_height], <br> <bf>x</bf>: [0, grid\_width], <br> *fire_reduction_power*: [0, initial\_fire\_power\_reduction], <br> <bf>suppressant</bf>: [0,suppressant\_states) <br> <br> <ins>Other Agents</ins> <br> <bf>y</bf>: [0,grid\_height], <br> <bf>x</bf>: [0, grid\_width], <br> *fire_reduction_power*: [0, initial\_fire\_power\_reduction], <br> <bf>suppressant</bf>: [0,suppressant\_states)  <br> <br> <ins>Task</ins> <br> <bf>y</bf>: [0,grid\_height], <br> <bf>x</bf>: [0, grid\_width], <br> <bf>fire level</bf>: [initial_fire_level] <br> *intensity*: [0,$`num\_fire\_states`$) |
|

[^1]:If `observable_suppressant` is true than observations include other agent's suppressant levels

[^2]: The second action value indicates the action taken for a specific task. Here there is only one action available for each task, but we include this to maintain a consistent form between environments. This is implemented with a **OneOf** space. 


[^2]: The second action value indicates the action taken for a specific task. Here there is only one action available for each task, but we include this to maintain a consistent form between environments. This is implemented with a **OneOf** space. 
