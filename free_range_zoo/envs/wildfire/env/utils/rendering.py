import random
from typing import Union, Optional
import os
import time
import math
from copy import deepcopy
from collections import defaultdict
from ast import literal_eval

import pygame
import numpy as np
import pandas as pd

# Adjust if you want to reference relative to this file
this_dir = os.path.dirname(__file__)

########################################################
#                IMAGE AND COLOR HELPERS              #
########################################################


def render_image(path, cell_size: int):
    """
    Loads an image from the local 'assets' folder,
    scales it to fit a given cell_size.
    """
    image = pygame.image.load(os.path.join(this_dir, "..", "assets", path))
    return pygame.transform.scale(image, (cell_size, cell_size))


def change_hue(image_surface, hue_change):
    """
    (Optional) Shifts the hue of a pygame.Surface by `hue_change` degrees.
    Returns a new surface with adjusted hue.
    """
    image_array = pygame.surfarray.array3d(image_surface).astype(np.float32)
    # Convert RGB to HSV
    r, g, b = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val

    hue = np.zeros_like(max_val)
    mask = delta != 0
    # Calculate the hue channel
    hue[mask & (max_val == r)] = (60 * ((g - b) / delta) + 0)[mask & (max_val == r)]
    hue[mask & (max_val == g)] = (60 * ((b - r) / delta) + 120)[mask & (max_val == g)]
    hue[mask & (max_val == b)] = (60 * ((r - g) / delta) + 240)[mask & (max_val == b)]
    hue = (hue + hue_change) % 360

    sat = np.zeros_like(max_val)
    sat[mask] = delta[mask] / max_val[mask]
    val = max_val / 255.0

    c = val * sat
    x = c * (1 - np.abs((hue / 60) % 2 - 1))
    m = val - c

    rr = np.zeros_like(hue)
    gg = np.zeros_like(hue)
    bb = np.zeros_like(hue)

    idx0 = (0 <= hue) & (hue < 60)
    idx1 = (60 <= hue) & (hue < 120)
    idx2 = (120 <= hue) & (hue < 180)
    idx3 = (180 <= hue) & (hue < 240)
    idx4 = (240 <= hue) & (hue < 300)
    idx5 = (300 <= hue) & (hue < 360)

    rr[idx0], gg[idx0], bb[idx0] = c[idx0], x[idx0], 0
    rr[idx1], gg[idx1], bb[idx1] = x[idx1], c[idx1], 0
    rr[idx2], gg[idx2], bb[idx2] = 0, c[idx2], x[idx2]
    rr[idx3], gg[idx3], bb[idx3] = 0, x[idx3], c[idx3]
    rr[idx4], gg[idx4], bb[idx4] = x[idx4], 0, c[idx4]
    rr[idx5], gg[idx5], bb[idx5] = c[idx5], 0, x[idx5]

    rr = ((rr + m) * 255).astype(np.uint8)
    gg = ((gg + m) * 255).astype(np.uint8)
    bb = ((bb + m) * 255).astype(np.uint8)

    new_image_array = np.stack([rr, gg, bb], axis=-1)
    new_surface = pygame.surfarray.make_surface(new_image_array)
    return new_surface


########################################################
#                   UI HELPERS: SLIDER                #
########################################################


def draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_position, max_time, t):
    """
    Draws a horizontal slider (gray bar + blue handle),
    updates t based on the position of the handle.
    """
    pygame.draw.rect(window, (150, 150, 150), (slider_x, slider_y, slider_width, slider_height))
    handle_x = slider_x + slider_position
    handle_x = slider_x + slider_position
    pygame.draw.rect(window, (0, 0, 255), (handle_x - 10, slider_y - 10, 20, 30))
    t = int((slider_position / slider_width) * max_time)
    return t


def draw_button(window, is_playing, button_x, button_y, button_size):
    """
    Draws a play/pause button to the right of the slider.
    """
    if is_playing:
        # Pause button (red square)
        pygame.draw.rect(window, (255, 0, 0), (button_x, button_y, button_size, button_size))
    else:
        # Play button (green triangle)
        pygame.draw.polygon(window, (0, 255, 0), [(button_x, button_y), (button_x, button_y + button_size),
                                                  (button_x + button_size, button_y + button_size // 2)])


def draw_time(window, t, screen_size, font):
    """
    Displays the current step/time near top-center of screen.
    """
    time_text = font.render(f"Step: {t}", True, (0, 0, 0))
    text_rect = time_text.get_rect(center=(screen_size // 2, 20))
    window.blit(time_text, text_rect)


########################################################
#            ARROW DRAWING FOR ANY VISUALS            #
########################################################


def find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset):
    """
    For drawing lines or arrows from one cell center to another.
    start_pos = (sx, sy), end_pos = (ex, ey) in grid coordinates (col, row).
    """
    sx, sy = start_pos
    ex, ey = end_pos
    start_center = (x_offset + sx * cell_size + cell_size // 2, y_offset + sy * cell_size + cell_size // 2)
    end_center = (x_offset + ex * cell_size + cell_size // 2, y_offset + ey * cell_size + cell_size // 2)
    return start_center, end_center


def draw_arrow(window, start_pos, end_pos, cell_size, x_offset, y_offset, use_water_effect=0):
    if use_water_effect == 0:
        """
        Draw a solid arrow from start cell to end cell in grid coordinates:
        start_pos = (sx, sy), end_pos = (ex, ey).
        """
        start_center, end_center = find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset)
        pygame.draw.line(window, (10, 50, 170), start_center, end_center, 5)

        # Calculate arrowhead
        angle = math.atan2(end_center[1] - start_center[1], end_center[0] - start_center[0])
        arrowhead_length = 15
        arrowhead_angle = math.radians(30)

        p1 = (end_center[0] - arrowhead_length * math.cos(angle - arrowhead_angle),
              end_center[1] - arrowhead_length * math.sin(angle - arrowhead_angle))
        p2 = (end_center[0] - arrowhead_length * math.cos(angle + arrowhead_angle),
              end_center[1] - arrowhead_length * math.sin(angle + arrowhead_angle))

        pygame.draw.polygon(window, (10, 50, 170), [end_center, p1, p2])

    else:
        """
        Simulates water spraying from a firefighter hose instead of a solid arrow.
        """
        start_center, end_center = find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset)

        # Water colors from light blue to white for realistic effect
        water_colors = [(173, 216, 230), (135, 206, 250), (200, 230, 255), (255, 255, 255)]  # Light Blue to White

        # Calculate main spray direction
        angle = math.atan2(end_center[1] - start_center[1], end_center[0] - start_center[0])

        # Main jet stream (center water flow)
        pygame.draw.line(window, (0, 191, 255), start_center, end_center, 6)  # Deep Sky Blue main stream

        # Generate multiple side sprays to simulate dispersion
        num_sprays = 20  # Adjust for more or fewer spray lines
        for _ in range(num_sprays):
            # Randomize deviation to simulate turbulence in water spray
            spray_angle = angle + random.uniform(-0.4, 0.4)  # Wider deviation
            spray_length = random.uniform(20, 80)  # Random spray distances

            spray_end = (start_center[0] + spray_length * math.cos(spray_angle),
                         start_center[1] + spray_length * math.sin(spray_angle))

            # Randomly select a water color
            spray_color = random.choice(water_colors)

            # Draw thin spray streams
            pygame.draw.line(window, spray_color, start_center, spray_end, random.randint(1, 3))

        # Water droplets forming mist around the spray
        for _ in range(30):  # More droplets for mist effect
            droplet_distance = random.uniform(10, 80)  # Random distance from hose exit
            droplet_angle = angle + random.uniform(-0.6, 0.6)  # Wider mist angle

            droplet_x = start_center[0] + droplet_distance * math.cos(droplet_angle)
            droplet_y = start_center[1] + droplet_distance * math.sin(droplet_angle)

            droplet_size = random.randint(2, 5)  # Small droplets
            droplet_color = random.choice(water_colors)  # Light blue or white
            pygame.draw.circle(window, droplet_color, (int(droplet_x), int(droplet_y)), droplet_size)

        # Water spray dispersion (wider at the end)
        spray_end_x = end_center[0] + random.uniform(-15, 15)  # Add slight spread
        spray_end_y = end_center[1] + random.uniform(-15, 15)
        pygame.draw.circle(window, (135, 206, 250), (int(spray_end_x), int(spray_end_y)), 8)  # Larger water impact


########################################################
#              MAIN RENDER FUNCTION                    #
########################################################


def render(path: str,
           render_mode: str = "human",
           frame_rate: Optional[int] = 15,
           checkpoint: Optional[int] = None) -> Union[None, list]:
    """
    Renders the wildfire environment from a single CSV log (path).

    Args:
        path        : path to a single CSV (like output/0.csv)
        render_mode : "human" or "rgb_array"
        frame_rate  : integer FPS if in "human" mode; if None => no throttle
        checkpoint  : if not None, filter steps by 'label' in CSV that match checkpoint

    Returns:
        None (if render_mode="human"), or
        a list of np.ndarray frames (if render_mode="rgb_array").
    """

    pygame.init()
    clock = pygame.time.Clock()
    df = pd.read_csv(path)

    # Convert certain columns from string to actual Python lists
    array_like_cols = [
        'fires',
        'intensity',
        'fuel',
        'suppressants',
        'capacity',
        'equipment',
        'agents',
    ]
    for col in array_like_cols:
        if col in df.columns:
            df[col] = df[col].fillna("[]")  # handle NaN
            df[col] = df[col].apply(lambda s: s.replace("nan", "[]") if isinstance(s, str) else s)
            df[col] = df[col].apply(literal_eval)

    # Agent action columns may or may not exist
    possible_agent_cols = [
        'firefighter_1_action',
        'firefighter_2_action',
        'firefighter_3_action',
    ]
    for col in possible_agent_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # If checkpoint is specified, filter the DataFrame
    if checkpoint is not None:
        df = df[df['label'] == checkpoint].reset_index(drop=True)
        if len(df) == 0:
            print(f"No steps found for label={checkpoint}")
            return None

    max_time = len(df) - 1
    if max_time < 0:
        print("No data to render. Exiting.")
        return None

    # Extract a name from the file path (for debug/UI)
    episode_name_str = os.path.basename(path)
    print(f"Episode: {episode_name_str}, Total steps: {max_time}")

    # ----------------------------------------------------------------
    # Infer the grid size from the first row's 'fires' data
    # `fires_grid_0` is a 2D array [ [ , , ], [ , , ] ]
    # row => y dimension, col => x dimension
    # ----------------------------------------------------------------
    fires_grid_0 = df['fires'].iloc[0]
    y = len(fires_grid_0)  # number of rows
    x = len(fires_grid_0[0]) if y > 0 else 0  # columns in each row

    cell_size = 190
    padding = 140
    grid_width = x * cell_size
    grid_height = y * cell_size
    screen_size = max(grid_width, grid_height) + padding * 2

    # If "human", we create a display; otherwise an offscreen Surface
    if render_mode == "human":
        window = pygame.display.set_mode((screen_size, screen_size + 150))
    else:
        window = pygame.Surface((screen_size, screen_size + 150))

    frames = []

    # ----------------------------------------------------------------
    # Pre-load images for each type of fire or agent
    # ----------------------------------------------------------------
    base_fire_low = render_image("small_fire.png", cell_size)
    base_fire_med = render_image("medium_fire.png", cell_size)
    base_fire_high = render_image("large_fire.png", cell_size)
    base_burnt = render_image("burnt_out.png", cell_size)
    base_agent = render_image("firefighter.png", cell_size)

    # ----------------------------------------------------------------
    # Precompute a row-major list of cell positions for "fire_number"
    # For example, if we have a grid:
    #   row=0 => cells 0..(x-1)
    #   row=1 => cells x..(2x-1)
    # etc.
    # ----------------------------------------------------------------
    fire_positions = []
    for fy, row in enumerate(fires_grid_0):
        for fx, fvalue in enumerate(row):
            if fvalue != 0:  # Fire exists
                fire_positions.append({
                    "fire": fx,  # Fire ID = column index
                    "y": fy,  # Row position
                    "x": fx,  # Column position
                    "intensity": fvalue  # Fire intensity
                })

    # ----------------------------------------------------------------
    # Build a per-step record of "objects" to render:
    #  - Fire cells with intensity/fuel
    #  - Agents with action
    # ----------------------------------------------------------------
    state_record = defaultdict(list)
    for i, row in df.iterrows():
        fires_2d = row['fires']
        intensity_2d = row['intensity']
        fuel_2d = row['fuel']
        agents_list = row['agents']

        # Build list of fire cell states
        for yy in range(y):
            for xx in range(x):
                intensity_val = intensity_2d[yy][xx]
                fuel_val = fuel_2d[yy][xx]
                cell_obj = {
                    "type": "fire",
                    "row": yy,
                    "col": xx,
                    "intensity": intensity_val,
                    "fuel": fuel_val,
                }
                state_record[i].append(cell_obj)

        # Build list of agent states
        for a_id, agent_pos in enumerate(agents_list):
            if not agent_pos or len(agent_pos) < 2:
                continue
            ay, ax = agent_pos  # (row, col) from CSV

            # Read the action from the CSV
            if a_id == 0:
                action_str = row['firefighter_1_action']
            elif a_id == 1:
                action_str = row['firefighter_2_action']
            else:
                action_str = row['firefighter_3_action']

            try:
                action_data = literal_eval(action_str) if isinstance(action_str, str) and action_str.strip() else []
            except:
                action_data = []

            # Also read the agent's suppressants, capacity, and reward
            sup_val = 0
            cap_val = 0
            if 'suppressants' in df.columns and a_id < len(row['suppressants']):
                sup_val = row['suppressants'][a_id]
            if 'capacity' in df.columns and a_id < len(row['capacity']):
                cap_val = row['capacity'][a_id]

            rw_col = f'firefighter_{a_id+1}_rewards'
            rew_val = row[rw_col] if rw_col in df.columns and pd.notna(row[rw_col]) else 0.0

            agent_obj = {
                "type": "agent",
                "id": a_id,
                "row": ay,
                "col": ax,
                "action": action_data,  # e.g. [firenumber, supp_power], or [0, -1] to "sleep"
                "suppressant": sup_val,
                "capacity": cap_val,
                "rewards": rew_val,
            }
            state_record[i].append(agent_obj)

    # ----------------------------------------------------------------
    # Initialize UI / controls
    # ----------------------------------------------------------------
    start_time = 0
    t = start_time
    slider_position = 0
    dragging_slider = False
    is_playing = False
    last_time = time.time()

    font = pygame.font.SysFont(None, 32)
    small_font = pygame.font.SysFont(None, 10)
    big_font = pygame.font.SysFont(None, 30)
    slider_width = 300
    slider_height = 10
    slider_x = (screen_size - slider_width) // 2
    slider_y = screen_size + 30
    button_size = 40
    button_x = slider_x + slider_width + 20
    button_y = slider_y - 15
    x_offset = (screen_size - grid_width) // 2
    y_offset = (screen_size - grid_height) // 2

    running = True
    while running:
        # -------------------- Handle events --------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if render_mode == "human":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if clicking on slider
                    if (slider_x <= event.pos[0] <= slider_x + slider_width) and \
                       (slider_y - 5 <= event.pos[1] <= slider_y + 15):
                        dragging_slider = True
                    # Check if clicking on play/pause button
                    if (button_x <= event.pos[0] <= button_x + button_size) and \
                       (button_y <= event.pos[1] <= button_y + button_size):
                        is_playing = not is_playing

                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_slider = False

                if event.type == pygame.MOUSEMOTION and dragging_slider:
                    slider_position = max(0, min(event.pos[0] - slider_x, slider_width))

        # -------------------- Auto-advance if playing --------------------
        if render_mode == "human":
            if is_playing and (time.time() - last_time >= 1.0) and not dragging_slider:
                last_time = time.time()
                if max_time > 0:
                    # Move the slider by one frame
                    slider_position = min(slider_width, slider_position + slider_width / max_time)
        else:
            # In rgb_array mode, move forward automatically each loop
            if max_time > 0:
                slider_position = min(slider_width, slider_position + slider_width / max_time)

        # Compute current time-step from slider
        t = int((slider_position / slider_width) * max_time)
        t = max(0, min(max_time, t))

        # -------------------- Clear screen --------------------
        window.fill((255, 255, 255))

        # -------------------- Draw grid lines --------------------
        line_color = (200, 200, 200)
        for row_i in range(y + 1):
            pygame.draw.line(window, line_color, (x_offset, y_offset + row_i * cell_size),
                             (x_offset + grid_width, y_offset + row_i * cell_size), 1)
        for col_i in range(x + 1):
            pygame.draw.line(window, line_color, (x_offset + col_i * cell_size, y_offset),
                             (x_offset + col_i * cell_size, y_offset + grid_height), 1)

        # -------------------- Draw slider / button / step info --------------------
        if render_mode == "human":
            draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_position, max_time, t)
            draw_button(window, is_playing, button_x, button_y, button_size)
        draw_time(window, t, screen_size, font)

        # Extra text: episode name, current step
        episode_info_text = f"Episode: {episode_name_str}  Step: {t}/{max_time}"
        episode_info_surf = small_font.render(episode_info_text, True, (0, 0, 0))
        window.blit(episode_info_surf, (slider_x, screen_size + 5))

        # -------------------- Render the state for this time-step --------------------
        fire_index = 0  # Just for labeling fires
        for obj in state_record[t]:
            cell_x = x_offset + obj["col"] * cell_size
            cell_y = y_offset + obj["row"] * cell_size
            if obj["type"] == "fire":
                # If intensity == 0, might skip rendering anything
                if obj["intensity"] == 0:
                    continue

                fire_tag = ''
                # Choose a base image by intensity
                if obj["intensity"] == 1:
                    base_img = base_fire_low
                    fire_tag = "Small Fire"
                    # scale_factor = 0.5
                elif obj["intensity"] == 2:
                    base_img = base_fire_med
                    fire_tag = "Medium Fire"
                    # scale_factor = 0.75
                elif obj["intensity"] == 3:
                    base_img = base_fire_high
                    fire_tag = "Large Fire"
                    # scale_factor = 1.0
                else:
                    # E.g. "4" or burnt
                    base_img = base_burnt
                    fire_tag = "Burnt Out"
                    # scale_factor = 0.75
                scale_factor = 0.9
                img_width = int(cell_size * scale_factor)
                img_height = int(cell_size * scale_factor)
                img_scaled = pygame.transform.scale(base_img, (img_width, img_height))

                center_x = cell_x + cell_size // 2
                center_y = cell_y + cell_size // 2
                draw_x = center_x - img_width // 2
                draw_y = center_y - img_height // 2
                window.blit(img_scaled, (draw_x, draw_y))

                # Optional text label
                fire_text = [
                    f"Fire {fire_index}",  # Fire index label
                    f"Intensity: {obj['intensity']} - {fire_tag}",
                    f"Fuel: {obj['fuel']}"
                ]
                fire_index += 1

                small_font = pygame.font.SysFont(None, 20)
                for idx, line in enumerate(fire_text):
                    line_surf = small_font.render(line, True, (0, 0, 0))
                    window.blit(line_surf, (cell_x + 5, cell_y + 5 + idx * 15))

            elif obj["type"] == "agent":
                scale_factor = 0.5
                img_width = int(cell_size * scale_factor)
                img_height = int(cell_size * scale_factor)
                img_scaled = pygame.transform.scale(base_agent, (img_width, img_height))

                center_x = cell_x + cell_size // 2
                center_y = cell_y + cell_size // 2
                draw_x = center_x - img_width // 2
                draw_y = center_y - img_height // 2
                window.blit(img_scaled, (draw_x, draw_y))

                # Show some textual info in the bottom part of the cell
                agent_text = [
                    f"Agent {obj['id']}", f"Suppressant: {obj['suppressant']}", f"Capacity: {obj['capacity']}",
                    f"Action: {obj['action']}", f"Reward: {obj['rewards']:.1f}"
                ]
                for idx, line in enumerate(agent_text):
                    line_surf = small_font.render(line, True, (0, 0, 0))
                    # Stack lines from the bottom upward
                    # window.blit(line_surf, (draw_x + 5, draw_y + cell_size - (len(agent_text) - idx) * 15))
                    window.blit(line_surf, (cell_x + 5, cell_y + 5 + idx * 15))

                # If the action is [fire_number, power], handle arrow or sleep
                if isinstance(obj["action"], (list, tuple)) and len(obj["action"]) == 2:
                    fire_num, power = obj["action"]  # e.g. [3, 2.0], or [0, -1]

                    # ---------------------------------------------------
                    # If [0, -1] => "no operation" or "sleep" => draw "Z"
                    # ---------------------------------------------------
                    if power == -1:
                        z_surf = big_font.render("NOOP", True, (250, 0, 0))
                        # Place the "Z" near the center of the agent's tile
                        # window.blit(z_surf, (draw_x + cell_size // 2, draw_y + cell_size // 2))
                        z_rect = z_surf.get_rect(center=(draw_x + 45, draw_y + img_height + 35))
                        window.blit(z_surf, z_rect)
                    # supress
                    if power == 0:

                        # If valid fire_number, draw arrow from agent to that fire
                        # if 0 <= fire_num <= len(fire_positions):
                        # Get the (row, col) of that fire from row-major list
                        fire_to_supress = next((fire for fire in fire_positions if fire.get("fire") == fire_num), None)
                        fire_row = fire_to_supress['y']
                        fire_col = fire_to_supress['x']

                        # Draw arrow from agent -> that fire cell
                        z_surf = big_font.render(f"Supress Fire {fire_num}", True, (0, 0, 250))
                        # Place the "Z" near the center of the agent's tile
                        # window.blit(z_surf, (draw_x + cell_size // 2, draw_y + cell_size // 2))
                        z_rect = z_surf.get_rect(center=(draw_x + 45, draw_y + img_height + 35))
                        window.blit(z_surf, z_rect)
                        draw_arrow(
                            window,
                            start_pos=(obj["col"], obj["row"]),  # (x, y) for agent
                            end_pos=(fire_col, fire_row),  # (x, y) for the target fire
                            cell_size=cell_size,
                            x_offset=x_offset,
                            y_offset=y_offset)

        # -------------------- Flip display or record frame --------------------
        if render_mode == "human":
            pygame.display.flip()
            if frame_rate is not None:
                clock.tick(frame_rate)
            else:
                clock.tick()  # no limit
        else:
            # Convert to array and store
            arr = pygame.surfarray.array3d(window)
            frames.append(arr)

            # If at last time-step, stop
            if t == max_time:
                running = False

        if not running:
            break

    # Done rendering
    pygame.quit()

    if render_mode == "rgb_array":
        return frames
    return None
