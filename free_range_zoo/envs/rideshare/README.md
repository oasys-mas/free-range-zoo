# Rideshare

Observations are always implemented with a **Tuple** space containing: a single `np.int` **Box** space for task-global observations, and multiple `np.int` **Box** spaces for task observations. **The exception** to this is if one of the observations is not a integer, in which case the observations spaces listed here are what is actually implemented. <ins>**This is one such environment**</ins>. We show the observation spaces not as **Box**es because it is easier to interprete that way. Please look at the `observation_space` function in each environment before using it. 

---

| Import             | `from freerangezoo.otask import rideshare_v0` |
|--------------------|------------------------------------|
| Actions            | Discrete \& Perfect                            |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agent Names             | ['$`driver\_0`$', ..., '$`driver\_n`$']` |
| #Agents             |    $`driver\_count`$                                  |
| Action Shape       | (2)                 |
| Action Values      | **Discrete**($`\|customers\|`$),**OneOf**( **Discrete**(1), $`\|customers\| \times`$ **Discrete**(1)  )                    |
| Observation Shape | (4,  8 $\times$ **Discrete**($`max\_customers`$)) |
| Observation Values   | <ins>Global Observation</ins> <br> *Agent Index*: **Discrete**($`driver\_count`$), <br> *Location*: **Discrete**($`grid\_x`$ * $`grid\_y`$), <br> *accepted_customers*: **Discrete**($\|customers\|$), <br> *passengers*: **Discrete**($\|customers\|$), <br> <br> <ins>Task Specific Observation</ins> <br>*rel_distance*: **Box**(low=0.0, high= $`grid\_max\_distance`$), <br> rel_direction: **Discrete**(4?**TODO** check direction options), <br> *agent_accepting*: **Discrete**($`driver\_count`$+1, start=-1), <br> *agent_riding_with*: **Discrete**($`driver\_count`$+1, start=-1), <br> <br> *start*: **Discrete**($`grid\_x`$ * $`grid\_y`$), <br> *destination*: **Discrete**($`grid\_x`$ * $`grid\_y`$), <br> *action_type*: **Discrete**(3), <br> *entry_time*: **Discrete**($`max\_timesteps`$)  |
 | 