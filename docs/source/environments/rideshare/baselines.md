# Baselines


### FIFO: [first_in_first_out](free_range_zoo/envs/rideshare/baselines/first_in_first_out.py)
<u>**FIFO: first_in_first_out**</u><br>
The agent will act on the longest waiting passenger. In a tie, it will choose the lowest index (task-focusing) passenger.

<u>**Reasoning**</u><br>
This is the same policy used by Uber in their ["FIFO Zones"](https://help.uber.com/en/driving-and-delivering/article/driving-at-the-airport?nodeId=1c8143ff-da32-46e6-b18d-254b9511efe0) for airport ride queuing. This works well when passengers enter the environment in similar locations and car capacity is low.  

### Greedy (task focused): [greedy_Tfocus](free_range_zoo/envs/rideshare/baselines/greedy_Tfocus.py)
<u>**Behavior**</u><br>
Calculate the distance required to pickup and dropoff all present passengers. Then accept, pickup, and dropoff the passenger with the minimum distance. Only considering one passenger at a time. 

<u>**Reasoning**</u><br>
This behavior is similar to [Greedy Matching](https://medium.com/gett-engineering/the-matching-magic-principles-of-matching-algorithms-in-on-demand-taxi-domain-f35365637dcf) for taxi dispatch. Whenever a passenger is available (the FIFO in that article) we choose the lowest ETA. This is a decent local optima when passengers enter/exit the environment around the same times, but it doesn't allow pooling.


### Greedy (task global): [greedy_Tglobal](free_range_zoo/envs/rideshare/baselines/greedy_Tglobal.py)
<u>**Behavior**</u><br>
Calculate the distnnce required to pickup and dropoff all present passengers. **Act** on the passenger with the shortest distance. This can pool. 

<u>**Reasoning**</u><br>
This policy is a good example of what can happen with a inefficient pooling policy. Its ability to succesfully deliver pooled passengers is dependent on their relative distances, and often it will switch between tasks rapidly without accomplishing many tasks. 


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


