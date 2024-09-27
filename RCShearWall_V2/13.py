import pygame
import pygame_gui
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Pocket Tanks-like Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)
GROUND_COLOR = (34, 139, 34)

# Tank properties
TANK_WIDTH, TANK_HEIGHT = 40, 20
CANNON_LENGTH = 30

# Game variables
gravity = 0.5
turn = 0  # 0 for player 1, 1 for player 2
scores = [0, 0]
game_over = False

# Create tanks
tank1 = pygame.Rect(50, HEIGHT - TANK_HEIGHT - 100, TANK_WIDTH, TANK_HEIGHT)
tank2 = pygame.Rect(WIDTH - 50 - TANK_WIDTH, HEIGHT - TANK_HEIGHT - 100, TANK_WIDTH, TANK_HEIGHT)

# Tank angles and power
angles = [45, 135]
powers = [50, 50]

# Create GUI manager
manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# Create sliders
angle_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((10, 10), (200, 20)),
    start_value=45,
    value_range=(0, 180),
    manager=manager
)

power_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((10, 40), (200, 20)),
    start_value=50,
    value_range=(10, 100),
    manager=manager
)

# Create labels
angle_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((220, 10), (100, 20)),
    text="Angle: 45",
    manager=manager
)

power_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((220, 40), (100, 20)),
    text="Power: 50",
    manager=manager
)


def draw_tank(tank, angle, color):
    pygame.draw.rect(screen, color, tank)
    end_pos = (
        tank.centerx + CANNON_LENGTH * math.cos(math.radians(angle)),
        tank.centery - CANNON_LENGTH * math.sin(math.radians(angle))
    )
    pygame.draw.line(screen, color, tank.center, end_pos, 5)


def shoot(start_pos, angle, power):
    time = 0
    positions = []
    while True:
        x = start_pos[0] + power * math.cos(math.radians(angle)) * time
        y = start_pos[1] - (power * math.sin(math.radians(angle)) * time - 0.5 * gravity * time ** 2)
        if x < 0 or x > WIDTH or y > HEIGHT:
            break
        positions.append((int(x), int(y)))
        time += 0.1
    return positions


def draw_projectile_path(positions):
    for pos in positions:
        pygame.draw.circle(screen, BLACK, pos, 3)
        pygame.display.flip()
        pygame.time.delay(20)


def draw_explosion(pos):
    colors = [RED, BLUE, WHITE, BLUE]  # Colors for the explosion
    for radius in range(1, 30):
        color = random.choice(colors)
        for _ in range(20):  # Draw multiple particles for each radius
            angle = random.uniform(0, 2 * math.pi)
            dx = int(radius * math.cos(angle))
            dy = int(radius * math.sin(angle))
            pygame.draw.circle(screen, color, (pos[0] + dx, pos[1] + dy), random.randint(1, 3))
        pygame.display.flip()
        pygame.time.delay(20)


def draw_terrain():
    pygame.draw.rect(screen, GROUND_COLOR, (0, HEIGHT - 100, WIDTH, 100))


def draw_scores():
    font = pygame.font.Font(None, 36)
    score1_text = font.render(f"Player 1: {scores[0]}", True, RED)
    score2_text = font.render(f"Player 2: {scores[1]}", True, BLUE)
    screen.blit(score1_text, (10, 70))
    screen.blit(score2_text, (WIDTH - 150, 70))


def show_game_over_screen():
    screen.fill(SKY_BLUE)
    font = pygame.font.Font(None, 74)
    if scores[0] > scores[1]:
        text = font.render("Player 1 Wins!", True, RED)
    else:
        text = font.render("Player 2 Wins!", True, BLUE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(3000)


# Game loop
clock = pygame.time.Clock()
running = True
shooting = False
while running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == angle_slider:
                    angles[turn] = int(event.value)
                    angle_label.set_text(f"Angle: {angles[turn]}")
                elif event.ui_element == power_slider:
                    powers[turn] = int(event.value)
                    power_label.set_text(f"Power: {powers[turn]}")

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not shooting and not game_over:
                shooting = True
                if turn == 0:
                    start_pos = (tank1.centerx + CANNON_LENGTH * math.cos(math.radians(angles[0])),
                                 tank1.centery - CANNON_LENGTH * math.sin(math.radians(angles[0])))
                else:
                    start_pos = (tank2.centerx + CANNON_LENGTH * math.cos(math.radians(angles[1])),
                                 tank2.centery - CANNON_LENGTH * math.sin(math.radians(angles[1])))

                positions = shoot(start_pos, angles[turn], powers[turn])
                draw_projectile_path(positions)
                hit_pos = positions[-1]

                # Check for hits
                if turn == 0 and tank2.collidepoint(hit_pos):
                    draw_explosion(hit_pos)
                    scores[0] += 1
                elif turn == 1 and tank1.collidepoint(hit_pos):
                    draw_explosion(hit_pos)
                    scores[1] += 1

                if max(scores) >= 3:  # Game ends when a player reaches 3 points
                    game_over = True
                else:
                    turn = 1 - turn
                shooting = False

        manager.process_events(event)

    manager.update(time_delta)

    if not game_over:
        # Draw everything
        screen.fill(SKY_BLUE)
        draw_terrain()
        draw_tank(tank1, angles[0], RED)
        draw_tank(tank2, angles[1], BLUE)
        draw_scores()

        manager.draw_ui(screen)

        pygame.display.flip()
    else:
        show_game_over_screen()
        running = False

pygame.quit()