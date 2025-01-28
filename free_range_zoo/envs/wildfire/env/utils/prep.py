from planning.experiments.configs.wildfire import setup3,single_agent_two_fires,single_agent_single_fire
import termcolor

printg = lambda x: print(termcolor.colored(x, "green"))
printr = lambda x: print(termcolor.colored(x, "red"))
printb = lambda x: print(termcolor.colored(x, "blue"))


from free_range_zoo.envs import wildfire_v0
import torch


env = wildfire_v0.parallel_env(
    max_steps=100,
    parallel_envs=2,
    configuration=setup3(),
    device=torch.device('cpu'),
    log_dir="outputs/wildfire_logging_test_0", 
)

obs = env.reset()
agents = env.agents

for i in range(10):
    print(f"Step {i}")
    actions = {agent: torch.tensor(env.action_space(agent).sample_nested()) for agent in agents}
    printr(f"Action taken: {actions}\n")
    obs, reward, term, trunc, info = env.step(actions)
    printb(f"\nObs:{obs}\n")
    printr(f"R:{reward}\n")
    printb(f"term/trunc:{term}/{trunc}\n")
    printr(f"info:{info}\n")
x=2