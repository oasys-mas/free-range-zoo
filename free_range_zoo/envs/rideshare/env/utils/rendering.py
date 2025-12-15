from typing import Union
import os, time, math
from copy import deepcopy
from collections import defaultdict
from ast import literal_eval

import pygame
import numpy as np
import pandas as pd

this_dir = os.path.dirname(__file__)


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
    width = 2
    # Draw the line (shaft of the arrow)
    pygame.draw.line(window, (0, 0, 0), start_edge, end_edge, width)

    # Calculate the angle of the arrow
    angle = math.atan2(end_edge[1] - start_edge[1], end_edge[0] - start_edge[0])

    # Define arrowhead size
    arrowhead_length = 15
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
    width = 2
    dashed_points(window, (0, 0, 0), start_edge, end_edge, width)

    # Draw the arrowhead (as before)
    angle = math.atan2(end_edge[1] - start_edge[1], end_edge[0] - start_edge[0])
    arrowhead_length = 15
    arrowhead_angle = math.radians(30)

    point1 = (end_edge[0] - arrowhead_length * math.cos(angle - arrowhead_angle),
              end_edge[1] - arrowhead_length * math.sin(angle - arrowhead_angle))
    point2 = (end_edge[0] - arrowhead_length * math.cos(angle + arrowhead_angle),
              end_edge[1] - arrowhead_length * math.sin(angle + arrowhead_angle))

    pygame.draw.polygon(window, (0, 0, 0), [end_edge, point1, point2])


def get_drivers_with_riders(row):
    """
    Given a row, returns a dict: driver_id -> list of passenger_ids
    """
    passengers = literal_eval(row['passengers']) if isinstance(row['passengers'], str) else row['passengers']
    riding_map = defaultdict(list)

    for p_idx, p_info in enumerate(passengers):
        if len(p_info) < 11:
            continue
        state = p_info[6]
        riding_with = p_info[7]
        picked_step = p_info[10]

        if state == 1 and riding_with != -1 and picked_step != -1:
            riding_map[riding_with].append(p_idx)

    return dict(riding_map)


def extract_render_data(row, num_drivers):
    """
    Extracts all relevant render data from a single CSV row.
    Returns a dictionary containing agents, passengers, accepted/riding maps, and driver actions.
    """
    row_data = {}

    # Parse agents and passengers
    # order y,x
    # agent Ids= index in the tensor
    agents = row['agents']
    passengers = row['passengers']
    row_data['agents'] = agents
    row_data['passengers'] = passengers

    # Collect accepted and riding passengers
    accepted_map = defaultdict(list)  # driver_id -> list of passenger indices
    riding_map = defaultdict(list)

    for p_idx, p_info in enumerate(passengers):
        if len(p_info) < 11:
            continue
        accepted_by = p_info[7]
        py, px = p_info[1], p_info[2]
        dy, dx = p_info[3], p_info[4]
        fare = p_info[5]
        state = p_info[6]  # accepted
        riding_with = p_info[7]
        picked_step = p_info[10]

        if accepted_by != -1:
            accepted_map[accepted_by].append(p_idx)
        if state == 1 and riding_with != -1 and picked_step != -1:
            riding_map[riding_with].append(p_idx)

    row_data['accepted_map'] = accepted_map
    row_data['riding_map'] = riding_map

    # Parse driver actions
    driver_actions = []
    for d_idx in range(num_drivers):
        col = f'driver_{d_idx+1}_action'
        if col not in row or pd.isnull(row[col]):
            driver_actions.append([-1, -1])
        else:
            try:
                driver_actions.append(literal_eval(row[col]))
            except:
                driver_actions.append([-1, -1])
    row_data['driver_actions'] = driver_actions

    return row_data


def colorize_passenger(base_img, hue_shift):
    return colorize_car(base_img, hue_shift)


def colorize_car(base_img, hue_shift):
    """
    Applies hue shift only to red car body pixels (255, 0, 0),
    and preserves the alpha channel (transparency).
    """
    img = base_img.copy()
    rgb_arr = pygame.surfarray.array3d(img).astype(np.uint8)
    alpha_arr = pygame.surfarray.array_alpha(img).astype(np.uint8)

    # Create mask for pure red pixels (car body)
    mask = (rgb_arr[:, :, 0] == 255) & (rgb_arr[:, :, 1] == 0) & (rgb_arr[:, :, 2] == 0)

    # Get red pixels to shift
    red_pixels = rgb_arr[mask].astype(np.float32) / 255.0
    r, g, b = red_pixels[:, 0], red_pixels[:, 1], red_pixels[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    hue = np.zeros_like(r)
    hue[delta != 0] = 0  # red = hue 0
    hue = (hue + hue_shift / 360.0) % 1.0

    c = maxc
    x = c * (1 - np.abs((hue * 6) % 2 - 1))

    r_, g_, b_ = np.zeros_like(hue), np.zeros_like(hue), np.zeros_like(hue)
    idx = (hue >= 0) & (hue < 1 / 6)
    r_[idx], g_[idx], b_[idx] = c[idx], x[idx], 0
    idx = (hue >= 1 / 6) & (hue < 2 / 6)
    r_[idx], g_[idx], b_[idx] = x[idx], c[idx], 0
    idx = (hue >= 2 / 6) & (hue < 3 / 6)
    r_[idx], g_[idx], b_[idx] = 0, c[idx], x[idx]
    idx = (hue >= 3 / 6) & (hue < 4 / 6)
    r_[idx], g_[idx], b_[idx] = 0, x[idx], c[idx]
    idx = (hue >= 4 / 6) & (hue < 5 / 6)
    r_[idx], g_[idx], b_[idx] = x[idx], 0, c[idx]
    idx = (hue >= 5 / 6) & (hue <= 1)
    r_[idx], g_[idx], b_[idx] = c[idx], 0, x[idx]

    # Convert to uint8 and put back in original RGB array
    rgb_arr[mask, 0] = (r_ * 255).astype(np.uint8)
    rgb_arr[mask, 1] = (g_ * 255).astype(np.uint8)
    rgb_arr[mask, 2] = (b_ * 255).astype(np.uint8)

    # Combine with original alpha channel
    surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surface, rgb_arr)
    pygame.surfarray.pixels_alpha(surface)[:, :] = alpha_arr

    return surface


# Replace hue-shifted cars with pre-colored images
def tint_surface(base_surface, color):
    """Apply a solid tint color to a surface while preserving alpha."""
    surface = base_surface.copy()
    arr = pygame.surfarray.pixels3d(surface)
    alpha = pygame.surfarray.pixels_alpha(surface)
    color_arr = np.array(color, dtype=np.uint8).reshape((1, 1, 3))
    mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)  # red-ish mask
    arr[mask] = color_arr
    pygame.surfarray.pixels_alpha(surface)[:, :] = alpha
    return surface


def render(path: str,
           render_mode: str = 'human',
           y=10,
           x=10,
           cell_size=50,
           line_color=(200, 200, 200),
           padding=50,
           frame_rate=30,
           checkpoint=None) -> Union[None, np.ndarray]:

    pygame.init()
    frames = []

    grid_w = x * cell_size
    grid_h = y * cell_size
    screen_size = max(grid_w, grid_h) + padding * 3

    # window = pygame.display.set_mode((screen_size, screen_size + 200))
    clock = pygame.time.Clock()

    df = pd.read_csv(path)
    df['agents'] = df['agents'].apply(literal_eval)

    driver_action_cols = [c for c in df.columns if c.startswith('driver_') and c.endswith('_action')]
    num_drivers = len(driver_action_cols)

    # Dynamic slider to agents

    extra_ui_height = 200 + num_drivers * (cell_size - 5)
    window = pygame.display.set_mode((screen_size, screen_size + extra_ui_height))

    df['passengers'] = df['passengers'].apply(literal_eval)

    # max_pass = len(df['passengers'])
    max_pass = max(len(p) for p in df['passengers'])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    base_car = pygame.image.load(os.path.join(this_dir, "..", "assets", "car_plain.png")).convert_alpha()
    base_car = pygame.transform.scale(base_car, (cell_size, cell_size))

    base_passenger = pygame.image.load(os.path.join(this_dir, "..", "assets", "passenger_plain.png")).convert_alpha()
    base_passenger = pygame.transform.scale(base_passenger, (cell_size, cell_size))
    small_pass = pygame.transform.scale(base_passenger, (cell_size // 3, cell_size // 3)).convert_alpha()

    location_black = pygame.image.load(os.path.join(this_dir, "..", "assets", "location_black.png")).convert_alpha()
    location_black = pygame.transform.scale(location_black, (cell_size // 2, cell_size // 2))

    location_green = pygame.image.load(os.path.join(this_dir, "..", "assets", "location_green.png")).convert_alpha()
    location_green = pygame.transform.scale(location_green, (cell_size // 2, cell_size // 2))

    high_contrast_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (0, 0, 0),  # Black
        (255, 165, 0),  # Orange
    ]

    # car_hue_step = 120 // max(1, num_drivers)
    pass_hue_step = 120 // max(1, max_pass)
    # Shift red body by these hue values
    hue_shifts = [0, 60, 120, 180, 240, 300]  # In degrees

    # car_assets = [colorize_car(base_car, hue_shift=hue_shifts[i % len(hue_shifts)]) for i in range(num_drivers)]
    car_assets = [tint_surface(base_car, high_contrast_colors[i % len(high_contrast_colors)]) for i in range(num_drivers)]
    # passenger_assets = [colorize_passenger(base_passenger, hue_shift=i * pass_hue_step) for i in range(max_pass)]
    passenger_assets = [
        tint_surface(base_passenger, high_contrast_colors[i % len(high_contrast_colors)]) for i in range(max_pass)
    ]
    # passenger_assets = [colorize_passenger(base_passenger, hue_shift=i * pass_hue_step) for i in range(max_pass)]
    small_pass_assets = [colorize_passenger(small_pass, hue_shift=i * pass_hue_step) for i in range(max_pass)]

    # ghost_passengers = [p.copy() for p in passenger_assets]
    # for gp in ghost_passengers:
    #     gp.set_alpha(128)

    pygame.font.init()
    font = pygame.font.SysFont(None, 32)
    tinyfont = pygame.font.SysFont(None, 16)

    max_time = df.shape[0] - 1
    t = 0
    slider_pos = 0
    dragging_slider = False
    is_playing = False
    last_time = time.time()

    slider_width = 300
    slider_height = 10
    slider_x = (screen_size - slider_width) // 2
    # slider_y = screen_size + 30

    slider_y = screen_size + extra_ui_height - 60  # near bottom of whole window
    button_size = 40
    button_x = slider_x + slider_width + 20
    button_y = slider_y - 15

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if slider_x <= event.pos[0] <= slider_x + slider_width and slider_y - 5 <= event.pos[1] <= slider_y + 15:
                    dragging_slider = True
                if button_x <= event.pos[0] <= button_x + button_size and button_y <= event.pos[1] <= button_y + button_size:
                    is_playing = not is_playing
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False
            elif event.type == pygame.MOUSEMOTION and dragging_slider:
                slider_pos = max(0, min(slider_width, event.pos[0] - slider_x))

        if is_playing and time.time() - last_time >= 1 and not dragging_slider:
            last_time = time.time()
            slider_pos = min(slider_width, slider_pos + slider_width / max_time)

        window.fill((255, 255, 255))

        x_off = (screen_size - grid_w) // 2
        y_off = (screen_size - grid_h) // 2

        for row_line in range(y + 1):
            pygame.draw.line(window, line_color, (x_off, y_off + row_line * cell_size),
                             (x_off + grid_w, y_off + row_line * cell_size), 1)
        for col_line in range(x + 1):
            pygame.draw.line(window, line_color, (x_off + col_line * cell_size, y_off),
                             (x_off + col_line * cell_size, y_off + grid_h), 1)

        if render_mode == 'human':
            t = draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_pos, max_time, t)
            draw_button(window, is_playing, button_x, button_y, button_size)
        else:
            t = min(t + 1, max_time)

        draw_title(window, checkpoint="", t=t, screen_size=screen_size, font=font)

        # === Legend ===
        legend_font = pygame.font.SysFont(None, 14)

        legend_y = y_off + grid_h + 15
        legend_cell_size = 20
        legend_x = x_off + 20
        spacing = 120  # horizontal space between items

        # ---------- First Line: Arrows ----------
        # Pickup (Dashed Arrow)
        draw_dash_arrow(window, (0, 0), (3, 0), legend_cell_size + 15, legend_x, legend_y)
        pickup_label = legend_font.render("Pickup", True, (0, 0, 0))
        window.blit(pickup_label, (legend_x, legend_y + 5))

        # Dropoff (Solid Arrow)
        dropoff_x = legend_x + spacing
        draw_arrow(window, (0, 0), (3, 0), legend_cell_size + 15, dropoff_x, legend_y)
        dropoff_label = legend_font.render("Dropoff", True, (0, 0, 0))
        window.blit(dropoff_label, (dropoff_x, legend_y + 5))

        # ---------- Second Line: Accepted / Unaccepted Destinations ----------
        legend_y2 = legend_y + 30  # move to second row

        # Unaccepted Destination (Black Icon)
        icon_black_x = legend_x
        window.blit(location_black, (icon_black_x, legend_y2))
        black_label = legend_font.render("= Unaccepted passenger destination", True, (0, 0, 0))
        window.blit(black_label, (icon_black_x + 30, legend_y2 + 15))

        # Accepted Destination (Colored Icon)
        icon_colored_x = icon_black_x + spacing + 30
        colored_icon = colorize_car(location_green.copy(), hue_shift=60)
        window.blit(colored_icon, (icon_colored_x + 75, legend_y2))
        colored_label = legend_font.render("= Accepted passenger destination", True, (0, 0, 0))
        window.blit(colored_label, (icon_colored_x + 100, legend_y2 + 15))
        # Horizontal line below the legend
        line_y = legend_y2 + 30  # Adjust vertical offset as needed
        pygame.draw.line(window, (0, 0, 0), (0, line_y), (window.get_width(), line_y), 1)

        raw_row = df.iloc[t]
        render_data = extract_render_data(raw_row, num_drivers)

        agents = render_data['agents']
        passengers = render_data['passengers']
        accepted_map = render_data['accepted_map']
        riding_map = render_data['riding_map']
        driver_actions = render_data['driver_actions']

        for d_idx, (driver_row, driver_col) in enumerate(agents):

            car_img = car_assets[d_idx] if d_idx < len(car_assets) else car_assets[-1]
            window.blit(car_img, (x_off + driver_col * cell_size, y_off + driver_row * cell_size))
            label_txt = tinyfont.render(f"Driver_{d_idx+1}", True, (0, 0, 0))
            window.blit(label_txt, (x_off + driver_col * cell_size, y_off + driver_row * cell_size))

            # Driver extra info
            icon_x = x_off  # space horizontally
            icon_y = y_off + grid_h + 75 + d_idx * (cell_size + 25)
            window.blit(car_img, (icon_x, icon_y))

            # Extract reward safely from current row
            reward_col = f'driver_{d_idx+1}_rewards'
            raw_reward = raw_row.get(reward_col, 0)
            reward = 0 if pd.isnull(raw_reward) or raw_reward is None else raw_reward

            # Get riding and accepted passengers
            riding_passengers = riding_map.get(d_idx, [])
            accepted_passengers = accepted_map.get(d_idx, [])
            accepted_only = [p for p in accepted_passengers if p not in riding_passengers]

            # Format strings
            riding_str = ", ".join([f"P{p}" for p in riding_passengers]) if riding_passengers else "None"
            accepted_str = ", ".join([f"P{p}" for p in accepted_only]) if accepted_only else "None"

            # Full display label
            info_text = f"Driver_{d_idx+1}: Reward {round(reward, 2)} | Riding: {riding_str} | Accepted: {accepted_str}"
            info_label = tinyfont.render(info_text, True, (0, 0, 0))
            window.blit(info_label, (icon_x + cell_size + 5, icon_y + cell_size // 4))

            offset_count = 0
            # if d_idx in accepted_map:
            #     for p_idx in accepted_map[d_idx]:
            #         if d_idx in riding_map and p_idx in riding_map[d_idx]:
            #             continue
            #         sp_img = small_pass_assets[p_idx] if p_idx < len(small_pass_assets) else small_pass_assets[-1]
            #         x_icon = x_off + 120 + offset_count * (cell_size // 2)
            #         y_icon = y_off + grid_h + 30 * d_idx + 15
            #         # window.blit(sp_img, (x_icon, y_icon))
            #         offset_count += 1

            # if d_idx in riding_map:
            #     riders_str = ", ".join([f"P{p}" for p in riding_map[d_idx]])
            #     rid_txt = tinyfont.render(f"Driver {d_idx} is RIDING: {riders_str}", True, (0, 0, 0))
            #     rx = x_off + 130
            #     ry = y_off + grid_h + 35 * d_idx
            #     window.blit(rid_txt, (rx, ry))

            d_action = driver_actions[d_idx]
            if d_action[1] != -1 and d_action[0] < len(passengers):
                target = passengers[d_action[0]]
                if len(target) >= 5:
                    py, px = target[1], target[2]
                    desty, destx = target[3], target[4]
                    if d_action[1] == 0:
                        draw_dash_arrow(window, (driver_col, driver_row), (px, py), cell_size, x_off, y_off)
                    elif d_action[1] == 1:
                        draw_arrow(window, (driver_col, driver_row), (destx, desty), cell_size, x_off, y_off)

        for p_idx, p_info in enumerate(passengers):
            if len(p_info) < 11:
                continue
            riding_with = p_info[7]
            picked_step = p_info[10]
            if riding_with != -1 and picked_step != -1:
                continue
            py, px = p_info[1], p_info[2]
            dy, dx = p_info[3], p_info[4]

            pass_img = passenger_assets[p_idx] if p_idx < len(passenger_assets) else passenger_assets[-1]
            # ghost_img = ghost_passengers[p_idx] if p_idx < len(ghost_passengers) else ghost_passengers[-1]
            window.blit(pass_img, (x_off + px * cell_size, y_off + py * cell_size))
            label_txt = tinyfont.render(f"P_{p_idx}", True, (0, 0, 0))
            window.blit(label_txt, (x_off + px * cell_size, y_off + py * cell_size))

            is_accepted = p_info[6]  # driver ID, -1 if unaccepted

            if is_accepted == 0:
                icon_to_draw = location_black
            else:
                icon_to_draw = location_green

            # Draw centered icon
            dest_x = x_off + dx * cell_size + (cell_size - icon_to_draw.get_width()) // 2
            dest_y = y_off + dy * cell_size + (cell_size - icon_to_draw.get_height()) // 2
            window.blit(icon_to_draw, (dest_x, dest_y))

            # dest_x = x_off + dx * cell_size + (cell_size - location_icon.get_width()) // 2
            # dest_y = y_off + dy * cell_size + (cell_size - location_icon.get_height()) // 2
            # window.blit(location_icon, (dest_x, dest_y))

            # window.blit(location_icon, (x_off + dx * cell_size, y_off + dy * cell_size))
            label_txt = tinyfont.render(f"P_D_{p_idx}", True, (0, 0, 0))
            window.blit(label_txt, (x_off + dx * cell_size, y_off + dy * cell_size))

        pygame.display.flip()

        if render_mode == 'rgb_array':
            frames.append(pygame.surfarray.array3d(window))
            if t >= max_time:
                running = False

        clock.tick(frame_rate if render_mode == 'human' else 0)

    pygame.quit()
    return frames if render_mode == 'rgb_array' else None