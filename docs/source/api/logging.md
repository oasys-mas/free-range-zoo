# Logging Guide

## Overview
Logging in Free-Range-Zoo enables tracking of environment states, which can be used for both debugging and rendering. This guide outlines how to use the built-in logging functionality, configure logging paths, and interpret logged data.

## Logging Setup
To enable logging, specify the `log_directory` parameter when initializing an environment. This directory will store log files containing state information for each independent environment instance. You can bypass this check by setting `override_initialization_check=True` on environment creation.

### Example Usage
To enable logging, construct an environment and pass the `log_directory` parameter:

```python
from free_range_zoo.envs import wildfire_v0

env = wildfire_v0.parallel_env(
    other_arguments...,  
    log_directory="your_path",
)
```

### Logging Directory Structure
- The specified `log_directory` **must not exist** before execution. If it does, an error will be thrown to prevent accidental overwrites.
- The logger will create a directory containing CSV log files named `0.csv`, `1.csv`, etc., corresponding to each parallel environment instance.
- We recommend organizing logs within an `outputs/` folder and naming experiments systematically, e.g., `outputs/wildfire_logs`.

#### Example Directory Layout
```
outputs/
├── wildfire_logging_test_0/  # Created automatically
│   ├── 0.csv                 # Log file for environment instance 0
│   ├── 1.csv                 # Log file for environment instance 1
│   └── ...
```

## Logged Data
Each log file contains the **state**, not the **observation**, meaning that in noisy environments, logs reflect the actual environment state rather than what agents perceive. Future updates will allow optional logging of observations where relevant.

## Logging Function Details
The `_log_environment` function is responsible for recording environment states. Below is an overview of its operation:

```python
def _log_environment(self, reset: bool = False, extra: pd.DataFrame = None) -> None:
```

### Parameters:
- `reset` *(bool)*: Indicates if the logging occurs due to an environment reset.
- `extra` *(pd.DataFrame, optional)*: Additional data to append to logs.

### Behavior:
- Converts the environment state to a pandas DataFrame.
- On reset, initializes action and reward columns.
- Logs agent actions, rewards, mappings, and step count.
- Appends additional data if provided.
- Saves each parallel environment instance as a separate CSV file.

## Alternative Logging with `logging.getLogger`
Custom logging can be achieved using Python's built-in `logging` module:

```python
import logging
main_logger = logging.getLogger(__name__)
main_logger.info("Custom log message")
```

This allows logging of custom information at any timestep.

## Integration with Rendering
The logged data is also used for rendering wildfire simulations. The `Wildfire Simulation Renderer` processes these logs to generate visualizations, including interactive and video-based outputs. See `rendering.py` files in `env/<environment name>/env/utils/` for details.

## Notes
- Ensure that the `log_directory` does not exist before execution.
- Logs reflect **environment states**, not agent observations.
- Logs are structured for compatibility with the rendering system.


