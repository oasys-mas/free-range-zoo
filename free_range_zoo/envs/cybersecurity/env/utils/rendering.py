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


def draw_aaline_arrow(window, color, start, end, width=3):
    """
    Draw a thicker arrow line from start -> end.
    """
    pygame.draw.line(window, color, start, end, width)

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrowhead_length = 12
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


class AgentGridPlacer:
    """
    Hidden grid system for placing agents on the outer edges based on node position.
    Nodes are classified as left/right/top/bottom based on their angle from center.
    Agents are placed directly aligned with their target node's position on the edge,
    with sub-slots for multiple agents targeting the same node.
    Defenders and attackers share the same grid to avoid overlapping.
    """
    def __init__(self, node_positions, center_x, center_y, screen_size, margin=-20):
        self.node_positions = node_positions
        self.center_x = center_x
        self.center_y = center_y
        self.screen_size = screen_size
        self.margin = margin
        
        # Classify each node as left, right, top, or bottom and store its actual coordinate
        self.node_sides = {}
        self.node_edge_coord = {}  # The coordinate along the edge (y for left/right, x for top/bottom)
        
        for idx, (nx, ny) in enumerate(node_positions):
            angle = math.atan2(ny - center_y, nx - center_x)
            deg = math.degrees(angle)
            
            # Classify side and store the relevant coordinate
            if -45 <= deg < 45:
                side = 'right'
                edge_coord = ny  # Use node's y position
            elif 45 <= deg < 135:
                side = 'bottom'
                edge_coord = nx  # Use node's x position
            elif deg >= 135 or deg < -135:
                side = 'left'
                edge_coord = ny  # Use node's y position
            else:  # -135 <= deg < -45
                side = 'top'
                edge_coord = nx  # Use node's x position
            
            self.node_sides[idx] = side
            self.node_edge_coord[idx] = edge_coord
        
        # Track occupied slots per node: {node_idx: set of sub_slot indices}
        # SHARED grid for both defenders and attackers
        self.occupied = {idx: set() for idx in range(len(node_positions))}
        
        # Separate grid for inactive agents (??? mode) at the bottom center
        self.inactive_slot = 0
    
    def reset(self):
        """Clear all occupied slots for a new frame."""
        for key in self.occupied:
            self.occupied[key].clear()
        self.inactive_slot = 0
    
    def get_inactive_position(self, agent_type):
        """
        Get a position for an inactive agent (??? mode) on a 2-row grid at the bottom.
        Uses 2 rows with good padding between agents.
        """
        slot = self.inactive_slot
        self.inactive_slot += 1
        
        # Layout: 2 rows, spread horizontally with good padding
        slot_spacing = 100  # Horizontal space between agents
        row_spacing = 20   # Vertical space between rows (doubled)
        agents_per_row = 6  # Max agents per row before wrapping
        
        # Determine row (0 = bottom row, 1 = second row from bottom)
        row = slot // agents_per_row
        col = slot % agents_per_row
        
        # Center the agents horizontally
        center_x = self.screen_size // 2
        row_width = (agents_per_row - 1) * slot_spacing
        start_x = center_x - row_width // 2
        
        x = start_x + col * slot_spacing
        
        # Y position: bottom row first, then second row above it
        bottom_margin = 35
        y = self.screen_size - bottom_margin - row * row_spacing
        
        return (x, y)
    
    def get_node_side(self, node_idx):
        """Get which side a node is on."""
        return self.node_sides.get(node_idx, 'right')
    
    def get_position(self, node_idx, agent_type):
        """
        Get a position for an agent directly aligned with the target node.
        Agent is placed on the edge corresponding to the node's side,
        at the same coordinate as the node (y for left/right, x for top/bottom).
        """
        side = self.node_sides.get(node_idx, 'right')
        edge_coord = self.node_edge_coord.get(node_idx, self.screen_size // 2)
        
        # Defenders are closer to center, attackers are at the edge
        if agent_type == 'defender':
            edge_offset = 55  # Distance from edge for defenders (inner)
        else:
            edge_offset = 55  # Distance from edge for attackers (outer) same for both for now,
        
        # Find an available sub-slot for this node
        occupied = self.occupied[node_idx]
        sub_slot = 0
        while sub_slot in occupied:
            sub_slot += 1
        occupied.add(sub_slot)
        
        # Sub-slot offset with padding (spread multiple agents targeting same node)
        slot_spacing = 100  # Padding between agents
        sub_offset = sub_slot * slot_spacing
        
        # Position agent on the edge, aligned with the node's coordinate
        if side == 'left':
            x = self.margin + edge_offset +30
            y = edge_coord + sub_offset
        elif side == 'right':
            x = self.screen_size - self.margin - edge_offset
            y = edge_coord + sub_offset
        elif side == 'top':
            x = edge_coord + sub_offset + 30
            y = self.margin + edge_offset + 10
        else:  # bottom
            x = edge_coord + sub_offset
            y = self.screen_size - self.margin - edge_offset
        
        return (x, y)


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


def _arrow_start_from_agent(agent_pos, target_pos, AGENT_RADIUS=25):
    """
    Helper that computes where an arrow should start from an agent's edge.
    """
    ax, ay = agent_pos
    tx, ty = target_pos
    dx = tx - ax
    dy = ty - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return (ax, ay)
    # Move start point AGENT_RADIUS pixels toward the target
    ratio = AGENT_RADIUS / dist
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
        total_nodes = max_node_index +1
    # else:
    # fallback if no network_state column => you can define a default or skip
    # total_nodes = 5  # Just some fallback; or read from somewhere else

    screen_size = 800  # 700 * 1.3 = 910 (30% bigger)
    bottom_ui_height = 150
    window_width = 900
    window_height = 900

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
    circle_radius = 260  # 200 * 1.3 = 260 (30% bigger)
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
    slider_y = screen_size + 70
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

        # draw_time(window, t, screen_size, font)

        info_text = f"Log: {episode_name_str} | Step: {t}/{max_time}"
        info_surf = small_font.render(info_text, True, (0, 0, 0))
        info_rect = info_surf.get_rect(center=(window_width // 2, slider_y - 30))
        window.blit(info_surf, info_rect)

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

            # Show latency - position based on node's side
            lat_surf = small_font.render(f"latency={latency}", True, (0, 0, 0))
            # Determine node side based on angle from center
            angle = math.atan2(ny - center_y, nx - center_x)
            deg = math.degrees(angle)
            lat_offset = 20  # Distance from node center
            
            if -45 <= deg < 45:  # Right side -> latency on left
                lat_rect = lat_surf.get_rect(midright=(nx - lat_offset, ny))
            elif 45 <= deg < 135:  # Bottom side -> latency on top
                lat_rect = lat_surf.get_rect(midbottom=(nx, ny - lat_offset))
            elif deg >= 135 or deg < -135:  # Left side -> latency on right
                lat_rect = lat_surf.get_rect(midleft=(nx + lat_offset, ny))
            else:  # Top side (-135 to -45) -> latency on bottom
                lat_rect = lat_surf.get_rect(midtop=(nx, ny + lat_offset))
            window.blit(lat_surf, lat_rect)

        # Separate defenders/attackers by name
        all_agent_names = sorted(agent_data.keys())
        defenders = [n for n in all_agent_names if n.startswith("defender_")]
        attackers = [n for n in all_agent_names if n.startswith("attacker_")]

        # Use grid placer to position agents on outer edges based on node sides
        grid_placer = AgentGridPlacer(node_positions, center_x, center_y, screen_size)
        grid_placer.reset()

        # Draw defenders
        for i, def_name in enumerate(defenders):
            d_info = agent_data[def_name]
            if not d_info["present"]:
                continue
            
            # Check if this defender has a valid action
            action_list = d_info["action"]
            has_valid_action = len(action_list) >= 2
            
            if has_valid_action:
                node_idx = d_info["location"]
                if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= total_nodes:
                    node_idx = 0
                # Get position on the outer edge corresponding to the node's side
                dx, dy = grid_placer.get_position(node_idx, 'defender')
            else:
                # Place inactive agents at the bottom
                dx, dy = grid_placer.get_inactive_position('defender')

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
            if has_valid_action:
                target_idx, code = action_list
                if isinstance(target_idx, int) and 0 <= target_idx < total_nodes:
                    tx, ty = node_positions[target_idx]
                    sx, sy = _arrow_start_from_agent((dx, dy), (tx, ty), AGENT_RADIUS=25)
                    ex, ey = _point_on_node_edge((tx, ty), (dx, dy), NODE_RADIUS=NODE_RADIUS)
                    draw_aaline_arrow(window, (0, 255, 0), (sx, sy), (ex, ey), width=3)

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
            has_valid_action = len(action_list) >= 2
            
            if has_valid_action:
                target_idx, code = action_list
                if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= total_nodes:
                    target_idx = 0
                # Get position on the outer edge corresponding to the target node's side
                ax, ay = grid_placer.get_position(target_idx, 'attacker')
            else:
                target_idx, code = (0, None)
                # Place inactive agents at the bottom
                ax, ay = grid_placer.get_inactive_position('attacker')

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
            if has_valid_action:
                a_str = action_name(code)
                tx, ty = node_positions[target_idx]
                sx, sy = _arrow_start_from_agent((ax, ay), (tx, ty), AGENT_RADIUS=25)
                ex, ey = _point_on_node_edge((tx, ty), (ax, ay), NODE_RADIUS=NODE_RADIUS)
                draw_aaline_arrow(window, (255, 0, 0), (sx, sy), (ex, ey), width=3)

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
