from typing import Union
import os, time, math
from copy import deepcopy
from collections import defaultdict
from ast import literal_eval

import pygame
import numpy as np
import pandas as pd

this_dir = os.path.dirname(__file__)

raise NotImplementedError("This was likely broken in the rewrite.")


def render_image(path, cell_size: int):
    """
    just handles pathing to the image
    """
    image = pygame.image.load(os.path.join(this_dir, "..", "assets", path))
    return pygame.transform.scale(image, (cell_size, cell_size))


def change_hue(image_surface, hue_change):
    """
    Shift the hue of the given image_surface by hue_change degrees.
    """
    image_array = pygame.surfarray.array3d(image_surface).astype(np.float32)

    # Convert RGB to HSV
    r, g, b = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val

    hue = np.zeros_like(max_val)
    mask = delta != 0

    # Compute hue
    hue[mask & (max_val == r)] = (60 * ((g - b) / delta) + 0)[mask & (max_val == r)]
    hue[mask & (max_val == g)] = (60 * ((b - r) / delta) + 120)[mask & (max_val == g)]
    hue[mask & (max_val == b)] = (60 * ((r - g) / delta) + 240)[mask & (max_val == b)]
    hue = (hue + hue_change) % 360  # shift hue

    sat = np.zeros_like(max_val)
    sat[mask] = delta[mask] / max_val[mask]
    val = max_val / 255.0

    c = val * sat
    x = c * (1 - np.abs((hue / 60) % 2 - 1))
    m = val - c

    # Reconstruct new r/g/b
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

    new_image_array = np.dstack([rr, gg, bb])
    new_surface = pygame.surfarray.make_surface(new_image_array)
    return new_surface.convert_alpha()


# Function to draw the slider and handle dragging
def draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_position, max_time, t):

    # Draw the slider bar
    pygame.draw.rect(window, (150, 150, 150), (slider_x, slider_y, slider_width, slider_height))

    # Draw the slider handle
    handle_x = slider_x + slider_position  # Move handle proportionally to slider_position
    pygame.draw.rect(window, (0, 0, 255), (handle_x - 10, slider_y - 10, 20, 30))

    # Update current time based on slider position (proportionally to max_time)
    t = int((slider_position / slider_width) * max_time)

    return t


# Function to draw the play/stop button
def draw_button(window, is_playing, button_x, button_y, button_size):
    if is_playing:
        # Draw stop button (square)
        pygame.draw.rect(window, (255, 0, 0), (button_x, button_y, button_size, button_size))
    else:
        # Draw play button (triangle)
        pygame.draw.polygon(window, (0, 255, 0), [(button_x, button_y), (button_x, button_y + button_size),
                                                  (button_x + button_size, button_y + button_size // 2)])


# Function to draw the current time at the top of the screen
def draw_title(window, checkpoint, t, screen_size, font):
    # time_text = font.render(f"Checkpoint: {checkpoint}, Time: {t}", True, (0, 0, 0))  # Render text
    time_text = font.render(f"Time: {t}", True, (0, 0, 0))  # Render text
    text_rect = time_text.get_rect(center=(screen_size // 2, 20))  # Centered at the top
    window.blit(time_text, text_rect)  # Draw the text


# Function to find the nearest edge or center based on adjacency
def find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset):
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    # Calculate center positions of both start and end cells
    start_center = (x_offset + start_x * cell_size + cell_size // 2, y_offset + start_y * cell_size + cell_size // 2)
    end_center = (x_offset + end_x * cell_size + cell_size // 2, y_offset + end_y * cell_size + cell_size // 2)

    # If the start and end cells are adjacent, use the center points for the arrow
    if abs(start_x - end_x) == 1 and start_y == end_y:  # Horizontal neighbors
        return start_center, end_center
    elif abs(start_y - end_y) == 1 and start_x == end_x:  # Vertical neighbors
        return start_center, end_center
    elif abs(start_x - end_x) == 1 and abs(start_y - end_y) == 1:  # Diagonal neighbors
        return start_center, end_center
    else:
        # For non-adjacent cells, keep the nearest edge logic
        direction = (end_center[0] - start_center[0], end_center[1] - start_center[1])

        nearest_start_edge = start_center
        nearest_end_edge = end_center

        if abs(direction[0]) > abs(direction[1]):
            # Horizontal direction
            if direction[0] > 0:
                nearest_start_edge = (x_offset + (start_x + 1) * cell_size, start_center[1])  # Right edge
                nearest_end_edge = (x_offset + end_x * cell_size, end_center[1])  # Left edge
            else:
                nearest_start_edge = (x_offset + start_x * cell_size, start_center[1])  # Left edge
                nearest_end_edge = (x_offset + (end_x + 1) * cell_size, end_center[1])  # Right edge
        else:
            # Vertical direction
            if direction[1] > 0:
                nearest_start_edge = (start_center[0], y_offset + (start_y + 1) * cell_size)  # Bottom edge
                nearest_end_edge = (end_center[0], y_offset + end_y * cell_size)  # Top edge
            else:
                nearest_start_edge = (start_center[0], y_offset + start_y * cell_size)  # Top edge
                nearest_end_edge = (end_center[0], y_offset + (end_y + 1) * cell_size)  # Bottom edge

        return nearest_start_edge, nearest_end_edge


# Function to draw an arrow from start to end
def draw_arrow(window, start_pos, end_pos, cell_size, x_offset, y_offset):
    start_edge, end_edge = find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset)

    # Draw the line (shaft of the arrow)
    pygame.draw.line(window, (0, 0, 0), start_edge, end_edge, 5)

    # Calculate the angle of the arrow
    angle = math.atan2(end_edge[1] - start_edge[1], end_edge[0] - start_edge[0])

    # Define arrowhead size
    arrowhead_length = 20
    arrowhead_angle = math.radians(30)

    # Calculate the points of the arrowhead
    point1 = (end_edge[0] - arrowhead_length * math.cos(angle - arrowhead_angle),
              end_edge[1] - arrowhead_length * math.sin(angle - arrowhead_angle))
    point2 = (end_edge[0] - arrowhead_length * math.cos(angle + arrowhead_angle),
              end_edge[1] - arrowhead_length * math.sin(angle + arrowhead_angle))

    # Draw the arrowhead
    pygame.draw.polygon(window, (0, 0, 0), [end_edge, point1, point2])


def dashed_points(surface, color, start_pos, end_pos, width=1, dash_length=10):
    # Calculate total line length
    total_length = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    # Number of dashes
    dashes = int(total_length / dash_length)
    # Direction of the line
    direction = ((end_pos[0] - start_pos[0]) / total_length, (end_pos[1] - start_pos[1]) / total_length)

    for i in range(dashes):
        if i % 2 == 0:  # Only draw every other segment
            start_dash = (start_pos[0] + direction[0] * i * dash_length, start_pos[1] + direction[1] * i * dash_length)
            end_dash = (start_pos[0] + direction[0] * (i + 1) * dash_length, start_pos[1] + direction[1] * (i + 1) * dash_length)
            pygame.draw.line(surface, color, start_dash, end_dash, width)


# Use this function to replace the solid line in `draw_arrow`:
def draw_dash_arrow(window, start_pos, end_pos, cell_size, x_offset, y_offset):
    start_edge, end_edge = find_arrow_points(start_pos, end_pos, cell_size, x_offset, y_offset)
    # Draw dashed line for the arrow shaft
    dashed_points(window, (0, 0, 0), start_edge, end_edge, 5)

    # Draw the arrowhead (as before)
    angle = math.atan2(end_edge[1] - start_edge[1], end_edge[0] - start_edge[0])
    arrowhead_length = 20
    arrowhead_angle = math.radians(30)

    point1 = (end_edge[0] - arrowhead_length * math.cos(angle - arrowhead_angle),
              end_edge[1] - arrowhead_length * math.sin(angle - arrowhead_angle))
    point2 = (end_edge[0] - arrowhead_length * math.cos(angle + arrowhead_angle),
              end_edge[1] - arrowhead_length * math.sin(angle + arrowhead_angle))

    pygame.draw.polygon(window, (0, 0, 0), [end_edge, point1, point2])


def render(
    path: str,
    render_mode: str,
    # Define grid dimensions (m rows, n columns)
    y=10,
    x=10,
    cell_size=50,  # Width and height of each grid cell
    line_color=(200, 200, 200),  # Gray color for the grid lines
    padding=50,  # Padding around the grid
    frame_rate=30,  # Frames per second (mostly matters for rgb_array rendering)
    checkpoint=None  #specific checkpoint to render
) -> Union[None, np.ndarray]:
    """
    
    """

    frames = []

    grid_width = y * cell_size
    grid_height = x * cell_size
    screen_size = max(grid_width, grid_height) + padding * 3  # Square screen size

    #?window configuration
    window = pygame.display.set_mode(size=(screen_size, screen_size + 200))

    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    assert render_mode in ['human', 'rgb_array'], "Invalid render mode. Choose from 'human' or 'rgb_array'."

    #?load log
    df = pd.read_csv(path)

    #filter by checkpoint if specified
    if checkpoint is not None:
        renderAll = False
        df = df[df['label'].apply(lambda s: f'{checkpoint}' in f'{s}')].reset_index(drop=True)
    else:
        renderAll = True  #maybe use this flag later

    df['locations'] = df['locations'].apply(literal_eval)
    df['destinations'] = df['destinations'].apply(literal_eval)

    #handle columns with NaNs
    df['associations'] = df['associations'].apply(lambda x: x.replace('nan', '-1'))
    df['timing'] = df['timing'].apply(lambda x: x.replace('nan', '-1'))

    df['associations'] = df['associations'].apply(literal_eval)
    df['timing'] = df['timing'].apply(literal_eval)

    #find agents
    agent_cols = df.columns[df.columns.str.contains('driver')]
    agent_states = [col for col in agent_cols if 'state' in col]
    agent_actions = [col for col in agent_cols if 'action_choice' in col]
    task_actions = [col for col in agent_cols if 'task-action-index' in col]

    for col in agent_cols:
        df[col] = df[col].apply(lambda x: x.replace('tensor', '')[1:-1] if type(x) == str else '[-1,-1]')
        df[col] = df[col].apply(literal_eval)

    number_of_agents = len(agent_states)
    max_number_of_tasks = df['locations'].apply(lambda x: len(x)).max()

    #preparing assets
    car_hues = 360 // number_of_agents
    passenger_hues = 360 // max_number_of_tasks
    # 2. Load base images and convert them for alpha
    base_car = pygame.image.load(os.path.join(this_dir, "..", "assets", "car_asset.png")).convert_alpha()
    base_passenger = pygame.image.load(os.path.join(this_dir, "..", "assets", "passenger_asset.png")).convert_alpha()
    small_passenger = pygame.transform.scale(base_passenger, (cell_size // 3, cell_size // 3)).convert_alpha()
    # Scale them to fit exactly in a cell
    base_car = pygame.transform.scale(base_car, (cell_size, cell_size))
    base_passenger = pygame.transform.scale(base_passenger, (cell_size, cell_size))

    # 3. Create hue variants for cars
    car_assets = []
    for i in range(number_of_agents):
        # Shift hue for each car
        car_surface = change_hue(base_car, 120)
        car_assets.append(car_surface)

    # 4. Create hue variants for passengers and their destinations
    passenger_assets = []
    small_passenger_assets = []
    passenger_dest_assets = []

    for i in range(max_number_of_tasks):
        # Shift hue for each passenger
        p_surface = change_hue(base_passenger, i * passenger_hues).convert_alpha()
        sp_surface = change_hue(small_passenger, i * passenger_hues).convert_alpha()

        # Make the destination a semi‐transparent copy of the passenger
        p_dest_surface = p_surface.copy()
        p_dest_surface.set_alpha(128)  # Adjust alpha for partial transparency

        passenger_assets.append(p_surface)
        small_passenger_assets.append(sp_surface)
        passenger_dest_assets.append(p_dest_surface)

    # 5. Store these assets in a dictionary for easy retrieval
    assets = {
        "car": car_assets,
        "passenger": passenger_assets,
        "smallpassenger": small_passenger_assets,
        "passenger_dest": passenger_dest_assets
    }
    assets = {
        'car': [change_hue(base_car, i * car_hues) for i in range(number_of_agents)],
        'smallpassenger': [change_hue(small_passenger, i * passenger_hues) for i in range(max_number_of_tasks)],
        'passenger': [change_hue(base_passenger, i * passenger_hues) for i in range(max_number_of_tasks)]
    }
    #create ghosts of destinations
    assets['passenger_dest'] = [v.copy().convert_alpha() for v in assets['passenger']]
    [img.set_alpha(128) for img in assets['passenger_dest']]

    #?organizing for rendering
    state_record = defaultdict(lambda *args, **kwargs: {})
    label_record = defaultdict(lambda *args, **kwargs: 'NaN')
    time_record = defaultdict(lambda *args, **kwargs: -1)

    for i, row in df.iterrows():
        time_step = {}
        first_timestep = i if row['label'] not in label_record.values() else list(label_record.values()).index(row['label'])
        label_record[i] = row['label']
        time_record[i] = first_timestep

        #add passengers (location and destination)
        for ix, passenger in enumerate(row['locations']):
            key = (passenger[1], passenger[2], ix + number_of_agents)
            dest_key = (row['destinations'][ix][1], row['destinations'][ix][2], ix + number_of_agents)
            time_step[key] = {
                'asset': assets['passenger'][ix],
                'smallasset': assets['smallpassenger'][ix],
                'type': 'passenger',
                'acceptedby': row['associations'][ix][1],
                'ridingwith': row['associations'][ix][2]
            }
            time_step[dest_key] = {'asset': assets['passenger_dest'][ix], 'type': 'passenger_dest'}
        state_record[i] = time_step

        for agent_index, (agent_state, agent_action) in enumerate(zip(agent_states, agent_actions)):

            ag_state = row[agent_state]
            ag_action = df.loc[i + 1][agent_action] if df.shape[0] > i + 1 else [-1, -1]
            #maps (agent local task, action) -> (global task "within this batch", action)
            # ag_task_action_map = df.loc[i + 1][task_actions[agent_index]] if df.shape[0] > i + 1 else None
            ag_task_action_map = row[task_actions[agent_index]]

            key = (ag_state[2], ag_state[3], agent_index)

            if ag_action[1] == -1:
                time_step[key] = {
                    'asset': assets['car'][agent_index],
                    'action': 'noop',
                    'name': f'driver_{agent_index}',
                    'type': 'driver'
                }
            else:
                #map the local task to the global task
                try:
                    ag_action = (ag_task_action_map[int(ag_action[0])][2] - 1, ag_action[1])
                except Exception as e:
                    # print(e)
                    print(f"Error mapping task index: {e}")
                    print(f"Current action: {ag_action}, Task-Action-Map: {ag_task_action_map}")
                    continue  # Skip this action if mapping fails
                # Ensure the index is valid
                if not (0 <= ag_action[0] < len(row['locations'])):
                    print(f"Invalid task index: {ag_action[0]}, Available tasks: {len(row['locations'])}")
                    print(f"Locations: {row['locations']}, Task-Action-Map: {ag_task_action_map}")
                    continue  # Skip this iteration instead of raising an error

                if ag_action[1] == 0:
                    time_step[key] = {
                        'asset': assets['car'][agent_index],
                        'action': 'accept',
                        'accept_position': row['locations'][int(ag_action[0])],
                        'name': f'driver_{agent_index}',
                        'type': 'driver'
                    }
                elif ag_action[1] in [1, 2]:
                    travel_dest = row['locations'][int(ag_action[0])] if i + 1 >= df.shape[0] else (df.loc[i + 1][agent_state][2],
                                                                                                    df.loc[i + 1][agent_state][3])
                    time_step[key] = {
                        'asset': assets['car'][agent_index],
                        'action': 'pickup' if ag_action[1] == 1 else 'dropoff',
                        'move': [travel_dest[1], travel_dest[2 - 1]],
                        'name': f'driver_{agent_index}',
                        'type': 'driver'
                    }
                else:
                    print(f"Invalid action index: {ag_action[1]}")
                    continue  # Skip invalid actions
                assert ag_action[0] < len(
                    row['locations']), f"Invalid task index: {ag_action[0]},\n {row['locations']},\n {ag_task_action_map}"

                if ag_action[1] == 0:
                    time_step[key] = {
                        'asset': assets['car'][agent_index],
                        'action': 'accept',
                        #TODO this won't be accurate because of 'global' vs 'local' indices
                        'accept_position': row['locations'][int(ag_action[0])],
                        'name': f'driver_{agent_index}',
                        'type': 'driver'
                    }

                elif ag_action[1] == 1 or ag_action[1] == 2:

                    if df.shape[0] < i + 1:
                        travel_dest = row['locations'][int(ag_action[0])]

                        time_step[key] = {
                            'asset': assets['car'][agent_index],
                            'action': 'pickup' if ag_action[1] == 1 else 'dropoff',
                            'move': [travel_dest[1], travel_dest[2]],
                            'name': f'driver_{agent_index}',
                            'type': 'driver'
                        }

                    else:
                        travel_dest = (df.loc[i + 1][agent_state][2], df.loc[i + 1][agent_state][3])

                        time_step[key] = {
                            'asset': assets['car'][agent_index],
                            'action': 'pickup' if ag_action[1] == 1 else 'dropoff',
                            'move': travel_dest,
                            'name': f'driver_{agent_index}',
                            'type': 'driver'
                        }
                else:
                    raise IndexError(f"Invalid action index: {ag_action[1]}")

    # state_record[0] = {(0, 0, 0): {'asset': assets['car'], 'move': (1, 3), 'name': 'driver_0', 'action': 'dropoff'}}

    # Calculate top-left corner for centering the grid
    x_offset = (screen_size - grid_width) // 2
    y_offset = (screen_size - grid_height) // 2

    #?rendering configuration
    start_time = 0
    max_time = df.shape[0] - 1
    t = start_time
    slider_position = t
    dragging_slider = False

    # Slider and Button setup
    slider_width = 300
    slider_height = 10
    slider_x = (screen_size - slider_width) // 2
    slider_y = screen_size + 30  # Below the grid
    button_size = 40
    button_x = slider_x + slider_width + 20
    button_y = slider_y - 15
    x_offset = (screen_size - grid_width) // 2
    y_offset = (screen_size - grid_height) // 2

    # Button state
    is_playing = False
    last_time = time.time()

    # Initialize Pygame font
    pygame.font.init()
    font = pygame.font.SysFont(None, 32)  # Use a default font, size 48
    tinyfont = pygame.font.SysFont(None, 24)  # Use a default font, size 24
    """
 
       ____                _           _             _ 
      |  _ \ ___ _ __   __| | ___ _ __(_)_ __   __ _| |
      | |_) / _ \ '_ \ / _` |/ _ \ '__| | '_ \ / _` | |
      |  _ <  __/ | | | (_| |  __/ |  | | | | | (_| |_|
      |_| \_\___|_| |_|\__,_|\___|_|  |_|_| |_|\__, (_)
                                               |___/   
 
    """

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle slider dragging
            if event.type == pygame.MOUSEBUTTONDOWN:
                if slider_x <= event.pos[0] <= slider_x + slider_width and slider_y - 5 <= event.pos[1] <= slider_y + 15:
                    dragging_slider = True

                # Handle button click
                if button_x <= event.pos[0] <= button_x + button_size and button_y <= event.pos[1] <= button_y + button_size:
                    is_playing = not is_playing  # Toggle play/stop state

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False

            if event.type == pygame.MOUSEMOTION and dragging_slider:
                slider_position = max(0, min(event.pos[0] - slider_x, slider_width))

        # Auto-increase slider position if playing
        if is_playing and time.time() - last_time >= 1 and not dragging_slider:
            last_time = time.time()
            slider_position = min(slider_width, slider_position + slider_width / max_time)

        #==============================================================
        # Fill the background (white in this case)
        window.fill((255, 255, 255))

        # Draw grid lines (vertical and horizontal)
        for _y in range(y + 1):  # Horizontal lines
            pygame.draw.line(
                window,
                line_color,
                (x_offset, y_offset + _y * cell_size),
                (x_offset + grid_width, y_offset + _y * cell_size),
                1  # Line thickness
            )
        for _x in range(x + 1):  # Vertical lines
            pygame.draw.line(
                window,
                line_color,
                (x_offset + _x * cell_size, y_offset),
                (x_offset + _x * cell_size, y_offset + grid_height),
                1  # Line thickness
            )
        #==============================================================

        #generate controls if human rendering
        if render_mode == 'human':
            t = draw_slider(window=window,
                            slider_x=slider_x,
                            slider_y=slider_y,
                            slider_width=slider_width,
                            slider_height=slider_height,
                            slider_position=slider_position,
                            max_time=max_time,
                            t=t)

            draw_button(window=window, is_playing=is_playing, button_x=button_x, button_y=button_y, button_size=button_size)
        else:
            t = t + 1 if t < max_time else max_time

        draw_title(window=window, checkpoint=label_record[t], t=t, screen_size=screen_size, font=font)

        #count offset for multiple passengers in driver/passenger list
        x_count_offset = {key: 0 for key in range(number_of_agents)}

        # Render each asset in the correct grid position
        for (_y, _x, order_in_df), asset_details in state_record[t].items():

            #determine offsets for multiple drivers in the same cell
            num_drivers = sum(
                [key[0] == _y and key[1] == _x for key in state_record[t].keys() if state_record[t][key]['type'] == 'driver'])
            if num_drivers > 1:
                scaled_driver_size = cell_size / num_drivers
                driver_offset = order_in_df * (scaled_driver_size // 2)
            else:
                scaled_driver_size = cell_size
                driver_offset = 0

            #handle nested offset for passengers only
            num_passengers = sum(
                [key[0] == _y and key[1] == _x for key in state_record[t].keys() if 'passenger' in state_record[t][key]['type']])

            if num_passengers > 1:
                scaled_passenger_size = cell_size / max(num_drivers, 1) / num_passengers
                passenger_offset = order_in_df * (scaled_passenger_size // 2)

            window.blit(asset_details['asset'],
                        (x_offset + _x * cell_size + driver_offset, y_offset + _y * cell_size + driver_offset))

            #handle identifying accepted passengers
            if asset_details['type'] == 'driver':
                agent_index = asset_details['name'].split('_')[-1]
                #pick position below grid * by number of agents
                labeling_text_pos = (cell_size * 2, y_offset + y * cell_size + 25 + 50 * int(agent_index))
                labeling_text = tinyfont.render(f"{asset_details['name']}", True, (0, 0, 0))
                window.blit(labeling_text, labeling_text_pos)

            if 'acceptedby' in asset_details and asset_details.get('acceptedby') != -1:
                agent_index = int(asset_details['acceptedby'])
                #pick position below grid * by number of agents
                labeling_image_pos = (cell_size * 4 + 40 * x_count_offset[agent_index],
                                      y_offset + y * cell_size + 25 + 50 * int(agent_index))
                x_count_offset[agent_index] += 1
                # print(labeling_image_pos, x_count_offset, agent_index, screen_size, asset_details['asset'])
                window.blit(asset_details['smallasset'], labeling_image_pos)

                if asset_details['ridingwith'] != -1:
                    riding_text = tinyfont.render(f"RIDING", True, (0, 0, 0))
                    window.blit(riding_text, (labeling_image_pos[0], labeling_image_pos[1] - 20))

            if 'move' in asset_details:
                # print(_y, _x, asset_details['move'])
                move_y, move_x = asset_details['move']
                if asset_details['action'] == 'pickup':
                    draw_dash_arrow(window, (_x, _y), (move_x, move_y),
                                    cell_size=cell_size,
                                    x_offset=x_offset + driver_offset,
                                    y_offset=y_offset + driver_offset)
                else:
                    draw_arrow(window, (_x, _y), (move_x, move_y),
                               cell_size=cell_size,
                               x_offset=x_offset + driver_offset,
                               y_offset=y_offset + driver_offset)

        # Update the display
        pygame.display.flip()

        #record frames if rgb_array
        if render_mode == 'rgb_array':
            frames.append(pygame.surfarray.array3d(window))
            if t == max_time:
                running = False

        if render_mode == 'human':
            clock.tick(frame_rate)
        else:
            clock.tick()

    # Quit Pygame
    pygame.quit()
    return frames if render_mode == 'rgb_array' else None
