# MOASEI FAQ

*have a question? Open a issue on our [git](https://github.com/oasys-mas/free-range-zoo/issues/new)*, or send a email to us (see MOASEI webpage for contact info).


## Competition Logistics

## Track #1 Cybersecurity

### How do the number of `patched`, `exploited`, and `vulnerable` states affect the simulation?

![Diagram a tensor split into three sections, patched -> vulnerable -> exploited](https://oasys-mas.github.io/free-range-zoo/docs/source/_static/img/CybersecurityDrawing_network_state_rewards.png)


These are three kinds of states that each node can be in. Nodes transition from state to state as defenders patch, and attackers exploit. This will shift nodes between these states, and accordingly shift rewards. 

> Note, Attacker recieve -1 * rewards, thus encouraging them to create deeply exploited nodes.


### What does `importance` mean?

![Diagram that shows importance being a multiple of the number of outgoing edges from nodes on the reward of that node](https://oasys-mas.github.io/free-range-zoo/docs/source/_static/img/cybersecurity_rewards.png)

`Importance` is a measure of the number of outgoing edges from each node. This multiplies the reward of that node, so a node with zero outgoing edges doesn't impact the rewards at all.


### What does `include_x` mean?

We use the parameter `include_x` to indicate whether `this` agent should observe `other` agents' attributes. For cybersecurity `presence`, and `power` determine whether we show if other agent's are present in the environment, and the network power of those agents in the `others` attribute of the observation. 

For the purposes of MOASEI 2025, use the `include_x` or `observe_other_x` as shown in [Environment Initialization](https://oasys-mas.github.io/free-range-zoo/events/mosaei-2025/environment_initialization.html).


## Track #2 Rideshare

## Track #3 Wildfire
