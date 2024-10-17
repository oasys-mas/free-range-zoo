import pygame
import os, time, math

from collections import defaultdict

# Initialize Pygame
pygame.init()

this_dir = os.path.dirname(__file__)


# Define grid dimensions (m rows, n columns)
m, n = 10, 10
cell_size = 50  # Width and height of each grid cell
line_color = (200, 200, 200)  # Gray color for the grid lines

# Define padding and square screen size
padding = 50  # Padding around the grid
grid_width = n * cell_size
grid_height = m * cell_size
screen_size = max(grid_width, grid_height) + padding * 2  # Square screen size


def render_image(path):
    image = pygame.image.load(os.path.join(this_dir, "..", "assets", path))
    return pygame.transform.scale(image, (cell_size, cell_size))


assets = {
    'car': render_image('car_asset.png'),
}

# Create the game window
window = pygame.display.set_mode((screen_size, screen_size+150))

# Calculate top-left corner for centering the grid
x_offset = (screen_size - grid_width) // 2
y_offset = (screen_size - grid_height) // 2



#TODO change to pandas df
state_record = defaultdict(lambda *args, **kwargs: {})
    
state_record[0] = {
        (0, 0): {'asset': assets['car'], 'move': (1, 3), 'name': 'driver_0', 'action': 'dropoff'},
}


start_time = 0
max_time = 10
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

# Button state
is_playing = False
last_time = time.time()

# Initialize Pygame font
pygame.font.init()
font = pygame.font.SysFont(None, 48)  # Use a default font, size 48
tinyfont = pygame.font.SysFont(None, 24)  # Use a default font, size 24


# Function to draw the slider and handle dragging
def draw_slider():
    global slider_position, t

    # Draw the slider bar
    pygame.draw.rect(window, (150, 150, 150), (slider_x, slider_y, slider_width, slider_height))

    # Draw the slider handle
    handle_x = slider_x + slider_position  # Move handle proportionally to slider_position
    pygame.draw.rect(window, (0, 0, 255), (handle_x - 10, slider_y - 10, 20, 30))

    # Update current time based on slider position (proportionally to max_time)
    t = int((slider_position / slider_width) * max_time)


# Function to draw the play/stop button
def draw_button():
    if is_playing:
        # Draw stop button (square)
        pygame.draw.rect(window, (255, 0, 0), (button_x, button_y, button_size, button_size))
    else:
        # Draw play button (triangle)
        pygame.draw.polygon(window, (0, 255, 0), [
            (button_x, button_y), 
            (button_x, button_y + button_size), 
            (button_x + button_size, button_y + button_size // 2)
        ])


# Function to draw the current time at the top of the screen
def draw_time():
    time_text = font.render(f"Time: {t}", True, (0, 0, 0))  # Render text
    text_rect = time_text.get_rect(center=(screen_size // 2, 20))  # Centered at the top
    window.blit(time_text, text_rect)  # Draw the text



# Function to find the nearest edge or center based on adjacency
def find_arrow_points(start_pos, end_pos):
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
def draw_arrow(start_pos, end_pos):
    start_edge, end_edge = find_arrow_points(start_pos, end_pos)

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
            start_dash = (start_pos[0] + direction[0] * i * dash_length,
                          start_pos[1] + direction[1] * i * dash_length)
            end_dash = (start_pos[0] + direction[0] * (i + 1) * dash_length,
                        start_pos[1] + direction[1] * (i + 1) * dash_length)
            pygame.draw.line(surface, color, start_dash, end_dash, width)

# Use this function to replace the solid line in `draw_arrow`:
def draw_dash_arrow(start_pos, end_pos):
    start_edge, end_edge = find_arrow_points(start_pos, end_pos)
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








# Set up the grid's dimensions and offsets
x_offset = (screen_size - grid_width) // 2
y_offset = (screen_size - grid_height) // 2


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
    for y in range(m + 1):  # Horizontal lines
        pygame.draw.line(
            window, line_color, 
            (x_offset, y_offset + y * cell_size), 
            (x_offset + grid_width, y_offset + y * cell_size), 
            1  # Line thickness
        )
    for x in range(n + 1):  # Vertical lines
        pygame.draw.line(
            window, line_color, 
            (x_offset + x * cell_size, y_offset), 
            (x_offset + x * cell_size, y_offset + grid_height), 
            1  # Line thickness
        )
    #==============================================================

    draw_slider()
    draw_button()
    draw_time()


    # Render each asset in the correct grid position
    for (y, x), asset_details in state_record[t].items():
        window.blit(asset_details['asset'], (x_offset + x * cell_size, y_offset + y * cell_size))
        if 'move' in asset_details:
            move_y, move_x = asset_details['move']
            if asset_details['action'] == 'pickup':
                draw_dash_arrow((y, x), (move_y, move_x))
            else:
                draw_arrow((y, x), (move_y, move_x))
    

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
