# Baselines

### noop
<u>**Behavior**</u><br>
The agent takes no action in all states, effectively leaving the environment unchanged where possible. If the
environment naturally evolves regardless of the agent's actions, the no-op policy simply observes without intervention.

<u>**Reasoning**</u><br>
The no-op policy serves as a baseline for understanding the impact of inaction in the environment. It highlights the
natural dynamics of the environment without any agent interference, providing a benchmark to compare active policies.
This policy is particularly useful in identifying whether external factors (e.g., environmental dynamics or other agents)
play a significant role in achieving rewards or whether deliberate actions are necessary for success.

### random
<u>**Behavior**</u><br>
The agent selects actions uniformly at random from the available action space, with no regard for the state, goals, or
consequences of the actions.

<u>**Reasoning**</u><br>
The random policy establishes a baseline for performance in the absence of any learning or strategy. It demonstrates the
environment's inherent difficulty by showing how likely success is when actions are chosen arbitrarily. This helps
evaluate the performance improvement of learned or more sophisticated policies over pure chance. It is especially valuable
in stochastic environments where outcomes may vary widely even with random actions.

### camp (defenders only)
<u>**Behavior**</u><br>
The defender agent moves to a node determined by the modulo of its agent index, ensuring an even distribution of agents
across all nodes in the environment. The defender continuously patches the nodes it is assigned to, reducing the
exploited state of those nodes. If the defender is disconnected from the environment (i.e., pulled out), it proceeds
to the node it was assigned, resuming its patching action upon reentry.

<u>**Reasoning**</u><br>
The defenders only policy focuses on minimizing the exploited state of the environment by spreading defenders evenly
across all nodes, ensuring that no single node becomes overly vulnerable. By continuously patching the nodes, the
defender prevents attackers from exploiting weaknesses and maintains the overall security of the system.

The modulo-based assignment ensures that the defenders do not concentrate on any one part of the environment too
heavily, which could result in vulnerabilities elsewhere. This strategy can be particularly effective in maintaining
a balanced defense across all nodes, especially when attackers target multiple nodes simultaneously.

The behavior of returning to a home node when disconnected and then resuming patching upon reentry emphasizes the
defender’s resilience and ensures that no nodes are left unpatched due to periodic disconnections. The continuous
patching action is a reliable way to prevent attackers from exploiting vulnerabilities for extended periods.

This policy is a baseline for understanding how a fixed and distributed defense mechanism can maintain system integrity,
even when facing attacks that may target different nodes. It can be used as a reference for evaluating more dynamic or
reactive strategies that adapt to the evolving attack patterns in the environment.

### patched (defender - greedy defensive)
<u>**Behavior**</u><br>
The defender agents following the patched policy prioritize fully patching the nodes closest to being fully patched.
Because the environment is partially observable, the defender may not always have full knowledge of the current patch
status of each node, so they reevaluate the status of the nodes every three timesteps. If the defender is disconnected
and returned to the home node, the target node is reset, and they immediately select a new target to patch before moving
to it.

<u>**Reasoning**</u><br>
The defender’s greedy defensive approach focuses on maximizing the impact of their actions by prioritizing nodes that
are nearest to full patching, as fully patched nodes are less likely to be exploited by attackers. The partial
observability of the environment means that defenders must periodically reevaluate their target nodes to ensure they
are focusing on the most critical vulnerabilities. The three-timestep reevaluation is a compromise between reaction
speed and stability in a dynamic environment.

The reset of the target upon return to the home node ensures that defenders do not waste time and effort on an outdated
or irrelevant target, especially after being disconnected. This behavior is designed to ensure that the defender can
always act with the most current information once re-entering the environment, even after a period of inaction.

This policy serves as a baseline for a consistent, if static, defense mechanism. It provides a foundation for
understanding the efficiency of defenders who are not reactive to attack actions but instead operate under the
assumption that patching nodes near completion will reduce vulnerabilities.

### patched (attacker - greedy offensive)
<u>**Behavior**</u><br>
The attacker agents following the patched policy prioritize attacking the most patched node. This is a greedy offensive
approach aimed at exploiting the defender’s tendency to focus on nodes that are close to being fully patched. Attackers
reevaluate their attack targets every three timesteps, using their full visibility of the environment (since they always
receive an observation of the state of all nodes) to adapt their strategy.

<u>**Reasoning**</u><br>
By attacking the most patched node, attackers exploit the defender's focus on partially patched nodes. These nodes are
often the most vulnerable because the defenders are expending significant resources on them, potentially leaving them
exposed to further attacks. This strategy capitalizes on the defender’s attention bias and can lead to significant
damage if the defenders are slow to react or reevaluate their patching targets.

The three-timestep reevaluation allows the attacker to adapt to the evolving defense strategy while maintaining
offensive pressure. Since attackers do not require a monitor action to reassess the environment, they can continuously
adapt to the defender’s behavior without waiting for specific feedback. This aggressive, opportunistic approach forces
defenders to continuously adapt their patching strategy to counteract the attacker’s focus on their most vulnerable
nodes.

This policy is useful for testing how well an attacker can exploit the defender's patching priorities and whether
focusing on high-value targets yields better results than a more randomized or spread-out attack strategy.

### exploited (defender - greedy offensive)
<u>**Behavior**</u><br>
The defender agents following the exploited policy prioritize patching the nodes that are the most exploited, meaning
they focus on nodes that have the highest exploited state. As with the patched policy, the defenders must periodically
reevaluate their target nodes every three timesteps due to the partial observability of the environment. If the defender
is disconnected and returned to the home node, the target is reset, and the defender immediately selects a new target
before moving to it.

<u>**Reasoning**</u><br>
The exploited policy emphasizes repairing the most vulnerable nodes, or those with the highest exploited state. This
approach aims to minimize the risk of attackers exploiting the most compromised nodes and stabilizing the environment
by addressing the most critical issues first. By patching the most exploited nodes, defenders can reduce the immediate
risk of further damage, especially in a dynamic environment where nodes can change rapidly due to attacks. The periodic
reevaluation allows defenders to adapt to new exploited states, ensuring that they are addressing the most urgent
threats as the environment evolves.

The reset of the target upon disconnection ensures that defenders always prioritize the most critical nodes when they
re-enter the environment, preventing them from wasting time on less exploited nodes. This behavior makes the defender
more responsive to immediate threats but still requires periodic assessment to handle the dynamic changes in the
environment.

### exploited (attacker - greedy defensive)
<u>**Behavior**</u><br>
The attacker agents following the exploited policy prioritize attacking the least exploited node, aiming to exploit
nodes that are the least targeted or have the lowest exploited state. Similar to the defenders, the attackers
reevaluate their target every three timesteps and continuously adapt their strategy. They do not need a monitor action
and can observe the environment fully, enabling them to track which nodes are being less actively defended.

<u>**Reasoning**</u><br>
The exploited policy for attackers is the reverse of the greedy offensive strategy in the patched policy. Instead of
attacking the most patched node, the attacker focuses on exploiting nodes that are less targeted, or those with the
lowest exploited state. This strategy aims to take advantage of any gaps in the defenders’ patching coverage. While
defenders may prioritize the most exploited nodes, attackers capitalize on the unpatched or under-patched nodes that
have been overlooked. By targeting the least exploited nodes, attackers can often secure easy victories, taking
advantage of the defender's limited resources and strategic focus.

The three-timestep reevaluation ensures that attackers can continuously adapt their strategies based on the changing
state of the environment. Without needing a monitor action, attackers can swiftly shift their attention to new nodes
that become vulnerable, making it difficult for defenders to maintain a consistent focus on all nodes.
