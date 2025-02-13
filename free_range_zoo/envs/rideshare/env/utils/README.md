# Rideshare Simulation Renderer

## Overview
This module provides a visualization tool for rideshare simulations. It processes a CSV log file containing simulation states and renders an interactive or video-based visualization using **Pygame**.

## Features
- **Renders rideshare simulations** based on logged agent states.
- **Supports two modes**:
  - `human`: Interactive visualization with a play/pause button and slider.
  - `rgb_array`: Generates frames for creating GIFs or videos.
- **Handles agent movements, interactions, and destinations**.
- **Color-coded cars and passengers** for better distinction.

## Dependencies
Ensure the following dependencies are installed:

```bash
pip install pygame numpy pandas imageio
```

## Usage

### 1. Running the Renderer
```python
import sys
import imageio
import numpy as np

sys.path.append('.')
from free_range_zoo.envs.rideshare.env.utils.rendering import render

csv_path = "path_to_your_log_file.csv"
render_mode = "human"  # Change to 'rgb_array' if you want frames
frame_rate = 15  # Frames per second
checkpoint = None  # Filter by label, if needed

render(path=csv_path, render_mode=render_mode, frame_rate=frame_rate, checkpoint=checkpoint)
```

### 2. Generating a GIF
```python
frames = render(csv_path, render_mode="rgb_array", frame_rate=500)
frames = [np.transpose(frame, (1, 0, 2)) for frame in frames]  # Adjust orientation

gif_path = "rideshare_simulation.gif"
imageio.mimsave(gif_path, frames, fps=1)
```

## API
```python
def render(
    path: str,
    render_mode: str,
    y=10, x=10,
    cell_size=50,
    line_color=(200, 200, 200),
    padding=50,
    frame_rate=30,
    checkpoint=None
) -> Union[None, np.ndarray]:
```
### Parameters:
- `path` *(str)*: Path to the CSV log file.
- `render_mode` *(str)*: Either `'human'` for interactive visualization or `'rgb_array'` for video frames.
- `y, x` *(int, optional)*: Grid dimensions (default: 10x10).
- `cell_size` *(int, optional)*: Size of each grid cell (default: 50px).
- `line_color` *(tuple, optional)*: RGB color of grid lines (default: light gray).
- `padding` *(int, optional)*: Padding around the grid (default: 50px).
- `frame_rate` *(int, optional)*: FPS for visualization (default: 30).
- `checkpoint` *(optional)*: Filter frames by label (default: None).

## Implementation Details
The renderer:
1. **Parses the CSV file** to extract agent states, destinations, and actions.
2. **Loads assets** (cars and passengers) and color-codes them for easy distinction.
3. **Renders each timestep** using Pygame, displaying agents, tasks, and movements.
4. **Provides interactivity**: Users can play, pause, and scrub through frames.
5. **Generates a sequence of frames** if using `rgb_array` mode, which can be converted into GIFs or videos.

## Notes
- Make sure the **CSV log file is correctly formatted**.
- Adjust **frame rates** accordingly for smooth playback.
- Ensure **Pygame window is properly closed** after execution.

## License
This project is licensed under MIT License.

## Contributors
Developed by Ali and the Free-Range-Zoo team.

