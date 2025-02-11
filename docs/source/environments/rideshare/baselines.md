# Baselines

### noop
<u>**Behavior**</u><br>
The agent takes no action in all states, effectively leaving the environment unchanged where possible. If the
environment naturally evolves regardless of the agent's actions, the no-op policy simply observes without intervention.

<u>**Reasoning**</u><br>
The no-op policy serves as a baseline for understanding the impact of inaction in the environment. It highlights the
natural dynamics of the environment without any agent interference, providing a benchmark to compare active policies.
This policy is particularly useful in identifying whether external factors (e.g., environmental dynamics or other
agents) play a significant role in achieving rewards or whether deliberate actions are necessary for success.

### random
<u>**Behavior**</u><br>
The agent selects actions uniformly at random from the available action space, with no regard for the state, goals, or
consequences of the actions.

<u>**Reasoning**</u><br>
The random policy establishes a baseline for performance in the absence of any learning or strategy. It demonstrates
the environment's inherent difficulty by showing how likely success is when actions are chosen arbitrarily. This helps
evaluate the performance improvement of learned or more sophisticated policies over pure chance. It is especially
valuable in stochastic environments where outcomes may vary widely even with random actions.

### baseline-1
<u>**Behavior**</u><br>
<--- INSERT HERE --->

<u>**Reasoning**</u><br>
<--- INSERT HERE --->

### baseline-2
<u>**Behavior**</u><br>
<--- INSERT HERE --->

<u>**Reasoning**</u><br>
<--- INSERT HERE --->
