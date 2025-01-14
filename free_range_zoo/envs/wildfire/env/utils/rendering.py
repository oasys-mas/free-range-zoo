from typing import Union, Optional
import os, time, math
from copy import deepcopy
from collections import defaultdict
from ast import literal_eval

import pygame
import numpy as np
import pandas as pd

this_dir = os.path.dirname(__file__)

########################################################
#                IMAGE AND COLOR HELPERS              #
########################################################


def render_image(path, cell_size: int):
    """
    Loads an image from the local 'assets' folder, 
    scales it to fit a given cell_size.
    """
    image = pygame.image.load(os.path.join(this_dir, "assets", path))
    return pygame.transform.scale(image, (cell_size, cell_size))


def change_hue(image_surface, hue_change):
    """
    Shifts the hue of a pygame.Surface by `hue_change` degrees.
    Returns a new surface with adjusted hue.
    """
    image_array = pygame.surfarray.array3d(image_surface).astype(np.float32)
    # Convert RGB to HSV
    r, g, b = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    # Initialize hue array
    hue = np.zeros_like(max_val)
    mask = delta != 0
    # Calculate the hue channel
    hue[mask & (max_val == r)] = (60 * ((g - b) / delta) + 0)[mask & (max_val == r)]
    hue[mask & (max_val == g)] = (60 * ((b - r) / delta) + 120)[mask & (max_val == g)]
    hue[mask & (max_val == b)] = (60 * ((r - g) / delta) + 240)[mask & (max_val == b)]
    hue = (hue + hue_change) % 360  # shift the hue
    # Saturation and Value
    sat = np.zeros_like(max_val)
    sat[mask] = delta[mask] / max_val[mask]
    val = max_val / 255.0
    # Reconstruct from HSV
    c = val * sat
    x = c * (1 - np.abs((hue / 60) % 2 - 1))
    m = val - c
    # Zero arrays for new r,g,b
    rr = np.zeros_like(hue)
    gg = np.zeros_like(hue)
    bb = np.zeros_like(hue)
    # Each hue slice
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
    # Stack and return surface
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
    pygame.draw.rect(window, (0, 0, 255), (handle_x - 10, slider_y - 10, 20, 30))
    t = int((slider_position / slider_width) * max_time)
    return t


def draw_button(window, is_playing, button_x, button_y, button_size):
    """
    Draws a play/pause button to the right of the slider.
    """
    if is_playing:
        pygame.draw.rect(window, (255, 0, 0), (button_x, button_y, button_size, button_size))
    else:
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
    """
    sx, sy = start_pos
    ex, ey = end_pos
    start_center = (x_offset + sx * cell_size + cell_size // 2, y_offset + sy * cell_size + cell_size // 2)
    end_center = (x_offset + ex * cell_size + cell_size // 2, y_offset + ey * cell_size + cell_size // 2)
    return start_center, end_center


def draw_arrow(window, start_pos, end_pos, cell_size, x_offset, y_offset):
    """
    Solid arrow from start cell to end cell.
    """
    start_edge, end_edge = find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset)
    pygame.draw.line(window, (0, 0, 0), start_edge, end_edge, 3)
    angle = math.atan2(end_edge[1] - start_edge[1], end_edge[0] - start_edge[0])
    arrowhead_length = 12
    arrowhead_angle = math.radians(30)
    p1 = (end_edge[0] - arrowhead_length * math.cos(angle - arrowhead_angle),
          end_edge[1] - arrowhead_length * math.sin(angle - arrowhead_angle))
    p2 = (end_edge[0] - arrowhead_length * math.cos(angle + arrowhead_angle),
          end_edge[1] - arrowhead_length * math.sin(angle + arrowhead_angle))
    pygame.draw.polygon(window, (0, 0, 0), [end_edge, p1, p2])


########################################################
#              MAIN RENDER FUNCTION                    #
########################################################


def render(path: str,
           render_mode: str = "human",
           frame_rate: Optional[int] = 15,
           checkpoint: Optional[int] = None) -> Union[None, np.ndarray]:
    """
    Renders the wildfire environment from a single CSV log (path).
    
    Args:
        path        : path to a single CSV (like output/0.csv)
        render_mode : "human" or "rgb_array"
        frame_rate  : integer FPS if in "human" mode, or None => no throttle
        checkpoint  : if not None, filter the steps by 'label' in CSV that match checkpoint

    Returns:
        None (if render_mode="human") 
        or 
        a list of np.ndarray frames (if render_mode="rgb_array").
    """

    ##########################################
    # 1. Initialize PyGame and read the CSV  #
    ##########################################
    pygame.init()
    clock = pygame.time.Clock()
    df = pd.read_csv(path)

    # Ensure array-like columns are processed.
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
            df[col] = df[col].fillna("[]")
            df[col] = df[col].apply(lambda s: s.replace("nan", "[]") if isinstance(s, str) else s)
            df[col] = df[col].apply(literal_eval)

    possible_agent_cols = [
        'firefighter_1_action_choice',
        'firefighter_2_action_choice',
        'firefighter_3_action_choice',
    ]
    for col in possible_agent_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    if checkpoint is not None:
        df = df[df['label'] == checkpoint].reset_index(drop=True)
        if len(df) == 0:
            print(f"No steps found for label={checkpoint}")
            return None

    max_time = len(df) - 1
    if max_time < 0:
        print("No data to render. Exiting.")
        return None

    # Extract and log the episode (CSV) filename.
    episode_name_str = os.path.basename(path)
    print(f"Episode: {episode_name_str}, Total steps: {max_time}")

    ##########################################
    # 2. Infer the grid size from the data   #
    ##########################################
    fires_grid_0 = df['fires'].iloc[0]
    y = len(fires_grid_0)  # rows
    x = len(fires_grid_0[0]) if y > 0 else 0  # columns

    cell_size = 190
    padding = 140
    grid_width = x * cell_size
    grid_height = y * cell_size
    screen_size = max(grid_width, grid_height) + padding * 2

    if render_mode == "human":
        window = pygame.display.set_mode((screen_size, screen_size + 150))
    else:
        window = pygame.Surface((screen_size, screen_size + 150))

    frames = []

    ##########################################
    # 3. Prepare image assets                #
    ##########################################
    base_fire_low = render_image("small_fire.png", cell_size)
    base_fire_med = render_image("medium_fire.png", cell_size)
    base_fire_high = render_image("large_fire.png", cell_size)
    # note that I dont own the burnt_out.png, just used it for now !
    # TODO
    base_burnt = render_image("burnt_out.png", cell_size)
    base_agent = render_image("firefighter.png", cell_size)

    ##########################################
    # 4. Build a structure of per-step info  #
    ##########################################
    state_record = defaultdict(list)
    for i, row in df.iterrows():
        fires_2d = row['fires']
        intensity_2d = row['intensity']
        fuel_2d = row['fuel']
        agents_list = row['agents']
        for yy in range(y):
            for xx in range(x):
                val = fires_2d[yy][xx]
                intensity_val = intensity_2d[yy][xx]
                fuel_val = fuel_2d[yy][xx]
                lit = (val != 0 and fuel_val > 0)
                cell_obj = {
                    "type": "fire",
                    "row": yy,
                    "col": xx,
                    "fire_type": abs(val),
                    "lit": lit,
                    "intensity": intensity_val,
                    "fuel": fuel_val,
                }
                state_record[i].append(cell_obj)
        for a_id, agent_pos in enumerate(agents_list):
            if not agent_pos or len(agent_pos) < 2:
                continue
            ay, ax = agent_pos
            if a_id == 0:
                action_str = row['firefighter_1_action_choice']
            elif a_id == 1:
                action_str = row['firefighter_2_action_choice']
            else:
                action_str = row['firefighter_3_action_choice']
            try:
                action_data = literal_eval(action_str) if isinstance(action_str, str) and action_str.strip() else []
            except:
                action_data = []
            sup_val = row['suppressants'][a_id] if a_id < len(row['suppressants']) else 0
            cap_val = row['capacity'][a_id] if a_id < len(row['capacity']) else 0
            rw_col = f'firefighter_{a_id+1}_rewards'
            rew_val = row[rw_col] if rw_col in df.columns and pd.notna(row[rw_col]) else 0.0
            agent_obj = {
                "type": "agent",
                "id": a_id,
                "row": ay,
                "col": ax,
                "action": action_data,
                "suppressant": sup_val,
                "capacity": cap_val,
                "rewards": rew_val,
            }
            state_record[i].append(agent_obj)

    ##########################################
    # 5. MAIN RENDER LOOP                    #
    ##########################################
    start_time = 0
    t = start_time
    slider_position = 0
    dragging_slider = False
    is_playing = False
    last_time = time.time()

    font = pygame.font.SysFont(None, 32)
    small_font = pygame.font.SysFont(None, 20)
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
        # ------------ Handle events ------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if render_mode == "human":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (slider_x <= event.pos[0] <= slider_x + slider_width) and (slider_y - 5 <= event.pos[1] <= slider_y + 15):
                        dragging_slider = True
                    if (button_x <= event.pos[0] <= button_x + button_size) and (button_y <= event.pos[1] <=
                                                                                 button_y + button_size):
                        is_playing = not is_playing
                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_slider = False
                if event.type == pygame.MOUSEMOTION and dragging_slider:
                    slider_position = max(0, min(event.pos[0] - slider_x, slider_width))
        # ------------ Step forward ------------
        if render_mode == "human":
            if is_playing and (time.time() - last_time >= 1.0) and not dragging_slider:
                last_time = time.time()
                if max_time > 0:
                    slider_position = min(slider_width, slider_position + slider_width / max_time)
        else:
            if max_time > 0:
                slider_position = min(slider_width, slider_position + slider_width / max_time)
        t = int((slider_position / slider_width) * max_time)
        t = max(0, min(max_time, t))
        # ------------ Clear screen -------------
        window.fill((255, 255, 255))
        # ------------ Draw grid -----------------
        line_color = (200, 200, 200)
        for row_i in range(y + 1):
            pygame.draw.line(window, line_color, (x_offset, y_offset + row_i * cell_size),
                             (x_offset + grid_width, y_offset + row_i * cell_size), 1)
        for col_i in range(x + 1):
            pygame.draw.line(window, line_color, (x_offset + col_i * cell_size, y_offset),
                             (x_offset + col_i * cell_size, y_offset + grid_height), 1)
        # ------------ Slider / Button / Step ---
        if render_mode == "human":
            draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_position, max_time, t)
            draw_button(window, is_playing, button_x, button_y, button_size)
        draw_time(window, t, screen_size, font)

        # ------------ Render Episode Information ------------
        # Display the episode name and step count (log file name) in the extra UI area.
        episode_info_text = f"Episode: {episode_name_str}  Step: {t}/{max_time}"
        episode_info_surf = small_font.render(episode_info_text, True, (0, 0, 0))
        window.blit(episode_info_surf, (slider_x, screen_size + 5))

        # ------------ Render all objects ------------
        fire_index = 0
        for obj in state_record[t]:
            if obj["type"] == "fire":
                # Always assign a unique fire name even if the fire isn't drawn.
                fire_name = f"F{fire_index}"
                fire_index += 1
                # If intensity is zero, skip drawing image/text.
                if obj["intensity"] == 0:
                    continue
                # Determine the image and scale factor based on intensity.
                if obj["intensity"] == 1:
                    base_img = base_fire_low
                    scale_factor = 0.5
                elif obj["intensity"] == 2:
                    base_img = base_fire_med
                    scale_factor = 0.75
                elif obj["intensity"] == 3:
                    base_img = base_fire_high
                    scale_factor = 1.0
                elif obj["intensity"] == 4:
                    base_img = base_burnt
                    scale_factor = 0.75
                # else:
                #     base_img = base_fire_high
                #     scale_factor = 1.0
                # Scale the image.
                img_width = int(cell_size * scale_factor)
                img_height = int(cell_size * scale_factor)
                img_scaled = pygame.transform.scale(base_img, (img_width, img_height))
                # Calculate the cell's position and center the scaled image.
                cell_x = x_offset + obj["col"] * cell_size
                cell_y = y_offset + obj["row"] * cell_size
                center_x = cell_x + cell_size // 2
                center_y = cell_y + cell_size // 2
                draw_x = center_x - img_width // 2
                draw_y = center_y - img_height // 2
                window.blit(img_scaled, (draw_x, draw_y))
                # Prepare and render overlay text.
                fire_text = [f"{fire_name}", f"intensity: {obj['intensity']:.1f}", f"Fuel: {obj['fuel']}"]
                for idx, line in enumerate(fire_text):
                    line_surf = small_font.render(line, True, (0, 0, 0))
                    window.blit(line_surf, (cell_x + 5, cell_y + 5 + idx * 15))
            elif obj["type"] == "agent":
                draw_x = x_offset + obj["col"] * cell_size
                draw_y = y_offset + obj["row"] * cell_size
                default_color = (0, 0, 0)
                agent_name = f"A{obj['id']}"
                window.blit(base_agent, (draw_x, draw_y))
                name_surf = small_font.render(agent_name, True, default_color)
                window.blit(name_surf, (draw_x + 5, draw_y + 5))
                agent_text = [
                    f"suppressant: {obj['suppressant']:.1f}", f"capacity: {obj['capacity']}", f"action: {obj['action']}",
                    f"reward: {obj['rewards']:.1f}"
                ]
                for idx, line in enumerate(agent_text):
                    line_surf = small_font.render(line, True, default_color)
                    window.blit(line_surf, (draw_x + 5, draw_y + cell_size - (len(agent_text) - idx) * 15))
        # ------------ Update display or record frame -----------
        if render_mode == "human":
            pygame.display.flip()
            if frame_rate is not None:
                clock.tick(frame_rate)
            else:
                clock.tick()
        else:
            arr = pygame.surfarray.array3d(window)
            frames.append(arr)
            if t == max_time:
                running = False
        if not running:
            break
    pygame.quit()
    if render_mode == "rgb_array":
        return frames
    return None
