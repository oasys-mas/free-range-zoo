import sys
import imageio
import numpy as np

sys.path.append('/home/alire/repos_wsl/free-range-zoo')
from free_range_zoo.envs.cybersecurity.env.utils.rendering import render

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "/home/alire/repos_wsl/free-range-zoo/outputs/cyberSec_logging_test_0/0.csv"

    # Render mode options: 'human' or 'rgb_array'
    render_mode = "rgb_array"  # Change to "rgb_array" if you want frames

    # Optional parameters
    frame_rate = 15  # Frames per second (None for as fast as possible)
    checkpoint = None  # Filter by label, if needed

    frames = render(csv_path, render_mode=render_mode, frame_rate=500)

    frames = [np.transpose(frame, (1, 0, 2)) for frame in frames]  # Swap width and height

    # frames = frames[::-1]  # Reverse the frames if they appear reversed in the GIF
    # Save frames as a GIF
    gif_path = "cybersecurity_simulation.gif"
    imageio.mimsave(gif_path, frames, fps=0.5)
    render_mode = "human"  # Change to "rgb_array" if you want frames

    # Optional parameters
    frame_rate = 15  # Frames per second (None for as fast as possible)
    checkpoint = None  # Filter by label, if needed

    # Call the renderer
    render(path=csv_path, render_mode=render_mode, frame_rate=frame_rate, checkpoint=checkpoint)
