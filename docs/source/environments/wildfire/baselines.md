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

### strongest
<u>**Behavior**</u><br>
The agent prioritizes fighting fires that are closest to burning out by focusing on tiles where the fire intensity is
highest. If the agent runs out of suppressant, it leaves the environment to refill its supply before returning to resume
operations.

<u>**Reasoning**</u><br>
In the wildfire-fighting domain, fires with high intensity are closer to reaching a tipping point where they can either
burn out naturally or intensify the spread to adjacent tiles. The strongest policy aims to strategically intervene by
suppressing these critical fires before they can escalate or self-extinguish. By targeting fires on the brink of burning
out, the agent maximizes its suppressant efficiency, potentially stabilizing the environment faster.

The policy's refill mechanism ensures the agent is always operational and avoids remaining idle when suppressant is
depleted. This behavior is crucial in scenarios where a delay in response could allow the fire to spread unchecked. The
strongest policy provides a benchmark for evaluating whether focusing on critical fires (as opposed to uniformly
suppressing all fires or prioritizing larger ones) is an effective strategy in mitigating the wildfire's overall impact.

### weakest
<u>**Behavior**</u><br>
The agent prioritizes fighting fires with the lowest intensity, focusing on tiles where the fire is closest to being
extinguished. If the agent runs out of suppressant, it leaves the environment to refill its supply before returning to
continue firefighting operations.

<u>**Reasoning**</u><br>
In the wildfire-fighting domain, extinguishing fires provides rewards, making it advantageous to target fires that are
closest to being put out. The weakest policy aims to harvest these rewards efficiently by allocating suppressant to
low-intensity fires, which require fewer resources to extinguish.

This strategy leverages the fact that fires close to extinguishment offer a high probability of success with minimal
effort. By concentrating on these easier targets, the agent maximizes short-term rewards while avoiding the risk of
wasting suppressant on more challenging, high-intensity fires. The refill mechanism ensures the agent remains
operational and ready to capitalize on emerging opportunities.

The weakest policy serves as a benchmark for evaluating reward-focused strategies, contrasting with other policies
that might prioritize long-term stability or overall fire suppression. It highlights the trade-offs between immediate
rewards and broader control of the environment. Baselines
