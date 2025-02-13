# Wildfire Simulation Renderer

## Overview
This module provides a visualization tool for wildfire simulations. It processes a CSV log file containing wildfire states and renders an interactive or video-based visualization using **Pygame**.
## A demo of the environment
<p align="center">
    <img src="assets/rendering.gif" width="400" height="400">
</p>


## Features
- **Renders wildfire simulations** based on logged fire intensity, agent actions, and environmental states.
- **Supports two modes**:
  - `human`: Interactive visualization with a play/pause button and slider.
  - `rgb_array`: Generates frames for creating GIFs or videos.
- **Handles fire intensity, suppression actions, and agent movements**.
- **Color-coded fire intensities and firefighter agents** for better distinction.

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

from rendering import render

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

gif_path = "wildfire_simulation.gif"
imageio.mimsave(gif_path, frames, fps=0.5)
```

## API
```python
def render(
    path: str,
    render_mode: str = "human",
    frame_rate: Optional[int] = 15,
    checkpoint: Optional[int] = None
) -> Union[None, list]:
```
### Parameters:
- `path` *(str)*: Path to the CSV log file.
- `render_mode` *(str)*: Either `'human'` for interactive visualization or `'rgb_array'` for video frames.
- `frame_rate` *(int, optional)*: FPS for visualization (default: 15).
- `checkpoint` *(optional)*: Filter frames by label (default: None).

## Implementation Details
The renderer:
1. **Parses the CSV file** to extract fire states, firefighter actions, and suppression attempts.
2. **Loads assets** (fire intensities, burnt areas, firefighter agents) and color-codes them for easy distinction.
3. **Renders each timestep** using Pygame, displaying fire spread and agent interactions.
4. **Provides interactivity**: Users can play, pause, and scrub through frames.
5. **Generates a sequence of frames** if using `rgb_array` mode, which can be converted into GIFs or videos.

## Notes
- Ensure the **CSV log file is correctly formatted** with necessary columns.
- Adjust **frame rates** accordingly for smooth playback.
- Ensure **Pygame window is properly closed** after execution.

## License
This project is licensed under MIT License.

## Contributors
Developed by Ali and the Free-Range-Zoo team.

