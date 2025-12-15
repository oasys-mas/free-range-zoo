# MOASEI FAQ

*Have a question? Open a issue on our
[GitHub](https://github.com/oasys-mas/free-range-zoo/issues/new)*, or send a email
to us (see MOASEI webpage for contact info).

## Competition Logistics

## Track #1 Cybersecurity

### How do the number of `patched`, `exploited`, and `vulnerable` states affect the simulation?

```{eval-rst}
.. image:: /_static/img/CybersecurityDrawing_network_state_rewards.png
   :alt: Diagram of a tensor split into three sections (patched -> vulnerable -> exploited)
```


These are three kinds of states that each node can be in. Nodes transition from
state to state as defenders patch, and attackers exploit. This will shift nodes
between these states, and accordingly shift rewards. 

> Note, Attacker recieve -1 * rewards, thus encouraging them to create deeply
> exploited nodes.


### What does `importance` mean?
```{eval-rst}
.. image:: /_static/img/cybersecurity_rewards.png
   :alt: Diagram that shows importance being a multiple of the number of outgoing edges from nodes on the reward of that node
```

`Importance` is a measure of the number of outgoing edges from each node. This
multiplies the reward of that node, so a node with zero outgoing edges doesn't
impact the rewards at all.

### What does `include_x` mean?

We use the parameter `include_x` to indicate whether `this` agent should
observe `other` agents' attributes. For cybersecurity `presence`, and `power`
determine whether we show if other agent's are present in the environment, and
the network power of those agents in the `others` attribute of the observation. 

For the purposes of MOASEI 2026, use the `include_x` or `observe_other_x` as
shown in [Environment
Initialization](https://oasys-mas.github.io/free-range-zoo/events/moasei-2026/environment_initialization.html).


## Track #2 Rideshare

## Track #3 Wildfire

### How does the equipment transition work for the bonus round?

In the wildfire environment, each agent's equipment can be in one of four
states, representing its condition:

- **Fully repaired (pristine)**
- **Intermediate state 1**
- **Intermediate state 2**
- **Fully damaged**

For the competition, only the *maximum suppressant* capacity of an agent is
affected by equipment state. The rules are as follows:

- **Fully repaired (pristine):** The agent's maximum suppressant is increased
  by 1 (+1 modifier).
- **Intermediate states (1 & 2):** The agent's maximum suppressant is unchanged
  (no modifier).
- **Fully damaged:** The agent's maximum suppressant is decreased by 1 (-1
  modifier).

Equipment state transitions are handled by the environment and occur
stochastically. Agents may degrade their equipment through use, suffer critical
errors, or repair their equipment when they perform the `noop` action.

This mechanism encourages agents to maintain their equipment in good condition
to maximize their firefighting effectiveness, while also introducing strategic
considerations around equipment management and resupply.
