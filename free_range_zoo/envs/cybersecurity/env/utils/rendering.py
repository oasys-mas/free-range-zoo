from typing import Union, Optional
import os
import time
import math
from ast import literal_eval

import pygame
import numpy as np
import pandas as pd
import re

this_dir = os.path.dirname(__file__)


def render_image(path: str, size: int):
    image = pygame.image.load(path)
    return pygame.transform.scale(image, (size, size))


def draw_aaline_arrow(window, color, start, end, width=2):
    """
    Draw a smoother (anti-aliased) arrow line from start -> end.
    """
    pygame.draw.aaline(window, color, start, end, True)

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrowhead_length = 10
    arrowhead_angle = math.radians(30)
    p1 = (end[0] - arrowhead_length * math.cos(angle - arrowhead_angle),
          end[1] - arrowhead_length * math.sin(angle - arrowhead_angle))
    p2 = (end[0] - arrowhead_length * math.cos(angle + arrowhead_angle),
          end[1] - arrowhead_length * math.sin(angle + arrowhead_angle))
    pygame.draw.polygon(window, color, [end, p1, p2])


def circular_layout(num_nodes, center_x, center_y, radius):
    coords = []
    for i in range(num_nodes):
        angle = (2 * math.pi * i) / max(1, num_nodes)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        coords.append((x, y))
    return coords


def distribute_angles(n_agents, total_angle_degs=40):
    """
    Distribute `n_agents` across -total_angle_degs/2..+total_angle_degs/2 (in degrees).
    Returns offsets in radians. Helps avoid overlapping agents.
    """
    if n_agents <= 1:
        return [0.0] * n_agents
    offsets = []
    total_rad = math.radians(total_angle_degs)
    step = total_rad / (n_agents - 1)
    start = -total_rad / 2
    for i in range(n_agents):
        offsets.append(start + i * step)
    return offsets


def action_name(action_value):
    """
    Numeric code => string action.
    0 => move, -1 => noop, -2 => patch, -3 => monitor, else => ???
    """
    if action_value == 0:
        return "move"
    elif action_value == -1:
        return "noop"
    elif action_value == -2:
        return "patch"
    elif action_value == -3:
        return "monitor"
    else:
        return "???"


def draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_position, max_time, t):
    pygame.draw.rect(window, (150, 150, 150), (slider_x, slider_y, slider_width, slider_height))
    handle_x = slider_x + slider_position
    pygame.draw.rect(window, (0, 0, 255), (handle_x - 10, slider_y - 10, 20, 30))
    if max_time > 0:
        t = int((slider_position / slider_width) * max_time)
    else:
        t = 0
    return t


def draw_button(window, is_playing, button_x, button_y, button_size):
    if is_playing:
        pygame.draw.rect(window, (255, 0, 0), (button_x, button_y, button_size, button_size))
    else:
        pygame.draw.polygon(window, (0, 255, 0), [(button_x, button_y), (button_x, button_y + button_size),
                                                  (button_x + button_size, button_y + button_size // 2)])


def draw_time(window, t, screen_size, font):
    time_text = font.render(f"Step: {t}", True, (0, 0, 0))
    text_rect = time_text.get_rect(center=(screen_size // 2, 20))
    window.blit(time_text, text_rect)


def _point_on_node_edge(node_center, agent_pos, NODE_RADIUS=20):
    """
    Helper that computes where an arrow should end on a node's circumference.
    """
    nx, ny = node_center
    ax, ay = agent_pos
    dx = nx - ax
    dy = ny - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return (nx, ny)
    ratio = (dist - NODE_RADIUS) / dist
    return (ax + ratio * dx, ay + ratio * dy)


def render(path: str,
           render_mode: str = "human",
           frame_rate: Optional[int] = 15,
           checkpoint: Optional[str] = None) -> Union[None, list]:

    pygame.init()
    clock = pygame.time.Clock()

    # Read CSV
    df = pd.read_csv(path)

    # A safer literal_eval that only converts bracketed strings to lists if possible.
    def safe_literal_eval_if_str(val):
        if isinstance(val, str):
            s = val.strip()
            # If it looks like a list (or tuple/dict), try literal_eval
            if (s.startswith("[") and s.endswith("]")) or \
               (s.startswith("(") and s.endswith(")")) or \
               (s.startswith("{") and s.endswith("}")):
                try:
                    return literal_eval(s)
                except:
                    pass
        return val

    # Dynamically parse columns that might be list-like
    # (For example, columns containing "action$", "presence", etc.)
    for col in df.columns:
        df[col] = df[col].apply(safe_literal_eval_if_str)

    # If there's a checkpoint label column and user wants to filter
    if checkpoint is not None and "label" in df.columns:
        df = df[df["label"] == checkpoint].reset_index(drop=True)
        if len(df) == 0:
            print(f"No rows found for label={checkpoint}")
            pygame.quit()
            return None

    max_time = len(df) - 1
    if max_time < 0:
        print("No data to render.")
        pygame.quit()
        return None

    episode_name_str = os.path.basename(path)
    # print(f"Episode: {episode_name_str}, total steps: {max_time}")
    print(f"Episode: {episode_name_str}, total steps: {max_time}")

    # Figure out how many nodes exist in the logs
    # We'll assume that if the "network_state" column exists, it might contain [env_id, node_idx, lat].
    total_nodes = 0
    if "network_state" in df.columns:
        max_node_index = 0
        for _, row in df.iterrows():
            ns = row["network_state"]
            # Expect [env_id, node_idx, latency], but only if well-formed
            if isinstance(ns, (list, tuple)) and len(ns) >= 3:
                node_idx = ns[1]
                if isinstance(node_idx, int) and node_idx > max_node_index:
                    max_node_index = node_idx
        total_nodes = max_node_index
    # else:
    # fallback if no network_state column => you can define a default or skip
    # total_nodes = 5  # Just some fallback; or read from somewhere else

    screen_size = 700
    bottom_ui_height = 120
    window_width = screen_size
    window_height = screen_size + bottom_ui_height

    if render_mode == "human":
        window = pygame.display.set_mode((window_width, window_height))
    else:
        window = pygame.Surface((window_width, window_height))

    frames = []

    # Try loading images
    try:
        node_img_exploited = render_image(os.path.join(this_dir, "..", "assets", "node_exploited.png"), 40)
        node_img_patched = render_image(os.path.join(this_dir, "..", "assets", "node_patched.png"), 40)
        node_img_normal = render_image(os.path.join(this_dir, "..", "assets", "node_normal.png"), 40)
    except:
        node_img_exploited = None
        node_img_patched = None
        node_img_normal = None

    try:
        attacker_img = render_image(os.path.join(this_dir, this_dir, "..", "assets", "attacker.png"), 40)
        defender_img = render_image(os.path.join(this_dir, this_dir, "..", "assets", "defender.png"), 40)
    except:
        attacker_img = None
        defender_img = None

    center_x = screen_size // 2
    center_y = screen_size // 2
    circle_radius = 200
    node_positions = circular_layout(total_nodes, center_x, center_y, circle_radius)

    # Identify any attacker/defender columns dynamically
    # e.g. attacker_1_action$, attacker_2_action$, ...
    attacker_cols = sorted([c for c in df.columns if re.match(r"attacker_\d+_action$", c)])
    defender_cols = sorted([c for c in df.columns if re.match(r"defender_\d+_action$", c)])

    def parse_presence(idx, presence_list):
        if idx < len(presence_list):
            return bool(presence_list[idx])
        return True

    # Build a time-indexed state record. This is the core structure for rendering.
    state_record = {}

    for t_i, row in df.iterrows():
        # Node-level info
        node_info = []
        # If exploited/patched columns exist, read them. Otherwise default to [False].
        if "exploited" in df.columns and isinstance(row["exploited"], (list, tuple)):
            exploited_list = row["exploited"]
        else:
            exploited_list = [False] * total_nodes

        if "patched" in df.columns and isinstance(row["patched"], (list, tuple)):
            patched_list = row["patched"]
        else:
            patched_list = [False] * total_nodes

        # Prepare node_info array
        for n_idx in range(total_nodes):
            e = bool(exploited_list[n_idx]) if n_idx < len(exploited_list) else False
            p = bool(patched_list[n_idx]) if n_idx < len(patched_list) else False
            node_info.append({"exploited": e, "patched": p, "latency": 0, "adj_matrix": row["adj_matrix"]})

        # If network_state is present, parse out node + latency
        if "network_state" in df.columns:
            ns = row["network_state"]
            if isinstance(ns, (list, tuple)) and len(ns) >= 3:
                # e.g. [env_id, node_idx, lat]
                the_node_idx = ns[1]
                lat = ns[2]
                if 0 <= the_node_idx < total_nodes:
                    node_info[the_node_idx]["latency"] = lat

        # If presence, location columns exist
        presence_list = row["presence"] if ("presence" in df.columns and isinstance(row["presence"], (list, tuple))) else []
        location_list = row["location"] if ("location" in df.columns and isinstance(row["location"], (list, tuple))) else []

        agents_info = {}

        # Parse defenders
        for idx, dcol in enumerate(defender_cols):
            # For example, dcol = "defender_1_action$"
            # Extract "defender_1"
            match_obj = re.match(r"(defender_\d+)_action$", dcol)
            if not match_obj:
                continue
            def_name = match_obj.group(1)

            # Is present or not?
            is_present = parse_presence(idx, presence_list)

            # Action might be a list: [target_node, action_code]
            def_action = row[dcol]
            if isinstance(def_action, str) and def_action.strip().upper() == "NULL":
                def_action = []
            elif not isinstance(def_action, (list, tuple)):
                def_action = []

            # Rewards for that defender if the column exists
            reward_col = def_name + "_rewards"
            def_reward = row.get(reward_col, 0.0)
            if isinstance(def_reward, str) and def_reward.strip().upper() == "NULL":
                def_reward = 0.0
            else:
                try:
                    def_reward = float(def_reward)
                except:
                    def_reward = 0.0

            # location for this defender
            d_loc = location_list[idx] if idx < len(location_list) else 0

            agents_info[def_name] = {"present": is_present, "location": d_loc, "action": def_action, "reward": def_reward}

        # Parse attackers
        for idx, acol in enumerate(attacker_cols):
            match_obj = re.match(r"(attacker_\d+)_action$", acol)
            if not match_obj:
                continue
            atk_name = match_obj.group(1)

            is_present = parse_presence(len(defender_cols) + idx, presence_list)

            atk_action = row[acol]
            if isinstance(atk_action, str) and atk_action.strip().upper() == "NULL":
                atk_action = []
            elif not isinstance(atk_action, (list, tuple)):
                atk_action = []
            reward_col = atk_name + "_rewards"

            atk_reward = row.get(reward_col, 0.0)
            if isinstance(atk_reward, str) and atk_reward.strip().upper() == "NULL":
                atk_reward = 0.0
            else:
                try:
                    atk_reward = float(atk_reward)
                except:
                    atk_reward = 0.0

            agents_info[atk_name] = {
                "present": is_present,
                "location": None,  # Attackers in your code appear to position themselves based on action target
                "action": atk_action,
                "reward": atk_reward
            }

        state_record[t_i] = {"nodes": node_info, "agents": agents_info}

    # Pygame UI setup
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
    slider_x = (window_width - slider_width) // 2
    slider_y = screen_size + 40
    button_size = 40
    button_x = slider_x + slider_width + 20
    button_y = slider_y - 15

    NODE_RADIUS = 20
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if render_mode == "human":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Slider click?
                    if (slider_x <= event.pos[0] <= slider_x + slider_width) and \
                       (slider_y - 5 <= event.pos[1] <= slider_y + 15):
                        dragging_slider = True
                    # Play/pause button click?
                    if (button_x <= event.pos[0] <= button_x + button_size) and \
                       (button_y <= event.pos[1] <= button_y + button_size):
                        is_playing = not is_playing

                if event.type == pygame.MOUSEBUTTONUP:
                    dragging_slider = False

                if event.type == pygame.MOUSEMOTION and dragging_slider:
                    slider_position = max(0, min(event.pos[0] - slider_x, slider_width))

        # Auto-advance in "human" mode if playing
        if render_mode == "human":
            if is_playing and (time.time() - last_time >= 1.0) and not dragging_slider:
                last_time = time.time()
                if max_time > 0:
                    slider_position = min(slider_width, slider_position + slider_width / max_time)
        else:
            # "rgb_array" mode: step automatically
            if max_time > 0:
                slider_position = min(slider_width, slider_position + slider_width / max_time)

        if max_time > 0:
            t = int((slider_position / slider_width) * max_time)
        else:
            t = 0
        t = max(0, min(max_time, t))

        window.fill((255, 255, 255))

        # Draw UI
        if render_mode == "human":
            t = draw_slider(window, slider_x, slider_y, slider_width, slider_height, slider_position, max_time, t)
            draw_button(window, is_playing, button_x, button_y, button_size)

        draw_time(window, t, screen_size, font)

        info_text = f"Episode: {episode_name_str} | Step: {t}/{max_time}"
        info_surf = small_font.render(info_text, True, (0, 0, 0))
        window.blit(info_surf, (slider_x, screen_size + 10))

        # Retrieve the current data from the stored record
        current_data = state_record[t]
        node_data = current_data["nodes"]
        agent_data = current_data["agents"]

        # Draw edges between nodes
        for n_idx, ninfo in enumerate(node_data):
            adj_matrix = ninfo.get("adj_matrix", [])
            for i in range(len(adj_matrix)):
                for j in range(i + 1, len(adj_matrix[i])):
                    if adj_matrix[i][j] != 0:
                        x1, y1 = node_positions[i]
                        x2, y2 = node_positions[j]
                        pygame.draw.aaline(window, (180, 180, 180), (x1, y1), (x2, y2), True)
        # Draw nodes
        for n_idx, ninfo in enumerate(node_data):
            nx, ny = node_positions[n_idx]
            exploited = ninfo.get("exploited", False)
            patched = ninfo.get("patched", False)
            latency = ninfo.get("latency", 0)

            # Pick an appropriate image if available
            if exploited and node_img_exploited:
                node_img = node_img_exploited
            elif patched and node_img_patched:
                node_img = node_img_patched
            else:
                node_img = node_img_normal

            # Fallback: draw a circle if no images
            if node_img:
                rect = node_img.get_rect(center=(nx, ny))
                window.blit(node_img, rect)
            else:
                color = (200, 200, 200)
                if exploited:
                    color = (255, 100, 100)
                elif patched:
                    color = (100, 200, 100)
                pygame.draw.circle(window, color, (int(nx), int(ny)), NODE_RADIUS)

            # Node label
            label_surf = small_font.render(f"{n_idx}", True, (0, 0, 0))
            label_rect = label_surf.get_rect(center=(nx, ny))
            window.blit(label_surf, label_rect)

            # Show latency
            lat_surf = small_font.render(f"latency={latency}", True, (0, 0, 0))
            window.blit(lat_surf, (nx - 25, ny + 22))

        # Separate defenders/attackers by name
        all_agent_names = sorted(agent_data.keys())
        defenders = [n for n in all_agent_names if n.startswith("defender_")]
        attackers = [n for n in all_agent_names if n.startswith("attacker_")]

        # Spread them around the circle
        def_angles = distribute_angles(len(defenders), total_angle_degs=15)
        atk_angles = distribute_angles(len(attackers), total_angle_degs=25)

        # Draw defenders
        for i, def_name in enumerate(defenders):
            d_info = agent_data[def_name]
            if not d_info["present"]:
                continue
            node_idx = d_info["location"]
            if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= total_nodes:
                node_idx = 0
            node_x, node_y = node_positions[node_idx]

            base_offset = 60
            angle = def_angles[i]
            angle_radians = math.atan2(node_y - center_y, node_x - center_x) + angle
            dx = center_x + (circle_radius + base_offset) * math.cos(angle_radians)
            dy = center_y + (circle_radius + base_offset) * math.sin(angle_radians)

            if defender_img:
                rect = defender_img.get_rect(center=(dx, dy))
                window.blit(defender_img, rect)
            else:
                pygame.draw.rect(window, (0, 0, 255), (dx - 20, dy - 20, 40, 40))

            # Name + Reward
            name_surf = small_font.render(def_name, True, (0, 0, 0))
            name_rect = name_surf.get_rect(midbottom=(dx, dy + 30))
            window.blit(name_surf, name_rect)

            rew_surf = small_font.render(f"Reward={d_info['reward']:.1f}", True, (0, 0, 0))
            rew_rect = rew_surf.get_rect(midtop=(dx, dy + 32))
            window.blit(rew_surf, rew_rect)

            # Action arrow
            action_list = d_info["action"]
            if len(action_list) >= 2:
                target_idx, code = action_list
                if isinstance(target_idx, int) and 0 <= target_idx < total_nodes:
                    tx, ty = node_positions[target_idx]
                    ex, ey = _point_on_node_edge((tx, ty), (dx, dy), NODE_RADIUS=NODE_RADIUS)
                    draw_aaline_arrow(window, (0, 255, 0), (dx, dy), (ex, ey), width=3)

                a_str = action_name(code)
                a_surf = small_font.render(a_str, True, (0, 255, 0))
                a_rect = a_surf.get_rect(midbottom=(dx, dy - 25))
                window.blit(a_surf, a_rect)
            else:
                a_surf = small_font.render("???", True, (0, 255, 0))
                a_rect = a_surf.get_rect(midbottom=(dx, dy - 25))
                window.blit(a_surf, a_rect)

        # Draw attackers
        for i, atk_name in enumerate(attackers):
            a_info = agent_data[atk_name]
            if not a_info["present"]:
                continue

            action_list = a_info["action"]
            if len(action_list) >= 2:
                target_idx, code = action_list
            else:
                target_idx, code = (0, None)

            if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= total_nodes:
                target_idx = 0

            node_x, node_y = node_positions[target_idx]
            base_offset = 25
            angle = atk_angles[i]
            angle_radians = math.atan2(node_y - center_y, node_x - center_x) + angle
            ax = center_x + (circle_radius + base_offset * 5) * math.cos(angle_radians)
            ay = center_y + (circle_radius + base_offset * 6) * math.sin(angle_radians)

            if attacker_img:
                rect = attacker_img.get_rect(center=(ax, ay))
                window.blit(attacker_img, rect)
            else:
                pygame.draw.rect(window, (255, 0, 0), (ax, ay, 40, 40))

            # Name + Reward
            name_surf = small_font.render(atk_name, True, (0, 0, 0))
            name_rect = name_surf.get_rect(midbottom=(ax, ay + 30))
            window.blit(name_surf, name_rect)

            rew_surf = small_font.render(f"Reward={a_info['reward']:.1f}", True, (0, 0, 0))
            rew_rect = rew_surf.get_rect(midtop=(ax, ay + 32))
            window.blit(rew_surf, rew_rect)

            # Action arrow + label
            if code is not None:
                a_str = action_name(code)
                tx, ty = node_positions[target_idx]
                ex, ey = _point_on_node_edge((tx, ty), (ax, ay), NODE_RADIUS=NODE_RADIUS)
                draw_aaline_arrow(window, (255, 0, 0), (ax, ay), (ex, ey), width=3)

                a_surf = small_font.render(a_str, True, (255, 0, 0))
                a_rect = a_surf.get_rect(midbottom=(ax, ay - 25))
                window.blit(a_surf, a_rect)
            else:
                a_surf = small_font.render("???", True, (255, 0, 0))
                a_rect = a_surf.get_rect(midbottom=(ax, ay - 25))
                window.blit(a_surf, a_rect)

        if render_mode == "human":
            pygame.display.flip()
            if frame_rate is not None:
                clock.tick(frame_rate)
        else:
            # Collect frames for "rgb_array" mode
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
