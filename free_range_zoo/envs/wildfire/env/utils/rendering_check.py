from rendering import render

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "/home/ali/repos/free-range-zoo/outputs/wildfire_logging_test_0/0.csv"

    # Render mode options: 'human' or 'rgb_array'
    render_mode = "rgb_array"  # Change to "rgb_array" if you want frames

    # Optional parameters
    frame_rate = 15  # Frames per second (None for as fast as possible)
    checkpoint = None  # Filter by label, if needed

    # Call the renderer
    render(
        path=csv_path,
        render_mode=render_mode,
        frame_rate=frame_rate,
        checkpoint=checkpoint
    )
    render_mode = "human"  # Change to "rgb_array" if you want frames

    # Optional parameters
    frame_rate = 15  # Frames per second (None for as fast as possible)
    checkpoint = None  # Filter by label, if needed

    # Call the renderer
    render(
        path=csv_path,
        render_mode=render_mode,
        frame_rate=frame_rate,
        checkpoint=checkpoint
    )