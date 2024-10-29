import torch


@torch.no_grad()
def chebyshev(agent_position: torch.Tensor, task_position: torch.Tensor, attack_range: torch.Tensor) -> torch.Tensor:
    """
    Checks if the task is within the attack range of the agent using Chebyshev distance

    Args:
        agent_position: torch.Tensor - vector of agent position(s)
        task_position: torch.Tensor - vector of task positions
        attack_range: torch.Tensor - vector of agent attack ranges

    Returns:
        torch.Tensor - boolean tensor indicating if the task is within the attack range of the agent
    """
    stacked_distances = torch.stack(
        [torch.abs(agent_position[:, 0] - task_position[:, 0]),
         torch.abs(agent_position[:, 1] - task_position[:, 1])], dim=1)

    chebyshev_distance = torch.max(stacked_distances, dim=1)[0]

    return chebyshev_distance <= attack_range


@torch.no_grad()
def euclidean(agent_position: torch.Tensor, task_position: torch.Tensor, attack_range: torch.Tensor) -> torch.Tensor:
    """
    Checks if the task is within the attack range of the agent using Euclidean distance

    Args:
        agent_position: torch.Tensor - vector of agent position(s)
        task_position: torch.Tensor - vector of task positions
        attack_range: torch.Tensor - vector of agent attack ranges

    Returns:
        torch.Tensor - boolean tensor indicating if the task is within the attack range of the agent
    """
    distance_diff = torch.stack(
        [torch.abs(agent_position[:, 0] - task_position[:, 0]),
         torch.abs(agent_position[:, 1] - task_position[:, 1])], dim=1)

    squared_diff_y = torch.pow(distance_diff[:, 0].float(), 2)
    squared_diff_x = torch.pow(distance_diff[:, 1].float(), 2)

    euclidean_distance = torch.sqrt(squared_diff_x + squared_diff_y)

    return euclidean_distance <= attack_range
