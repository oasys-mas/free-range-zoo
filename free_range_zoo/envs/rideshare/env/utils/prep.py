import sys
import termcolor
import sys
from pathlib import Path
import sys
from pathlib import Path

p = Path(__file__).resolve()
while p.name != "dev-free-range-zoo":
    p = p.parent
sys.path.insert(0, str(p))
from tests.utils import rideshare_configs

printg = lambda x: print(termcolor.colored(x, "green"))
printr = lambda x: print(termcolor.colored(x, "red"))
printb = lambda x: print(termcolor.colored(x, "blue"))

from free_range_zoo.envs import rideshare_v0

import torch

configuration = rideshare_configs.non_stochastic()

env = rideshare_v0.parallel_env(
    max_steps=40,
    render_mode='human',
    parallel_envs=2,
    configuration=rideshare_configs.non_stochastic(),
    device=torch.device('cpu'),
    log_directory="outputs/rideshare_logging_test_0",
)

obs = env.reset()
agents = env.agents

for i in range(40):
    print(f"Step {i}")
    actions = {agent: torch.tensor(env.action_space(agent).sample_nested()) for agent in agents}
    printr(f"Action taken: {actions}\n")
    obs, reward, term, trunc, info = env.step(actions)
    printb(f"\nObs:{obs}\n")
    printr(f"R:{reward}\n")
    printb(f"term/trunc:{term}/{trunc}\n")
    printr(f"info:{info}\n")
x = 2
