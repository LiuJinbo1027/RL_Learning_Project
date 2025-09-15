# space_invaders.py
# This is the base Pygame game for the space invaders-like game.
# It can be run standalone to play manually.

import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Invaders")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Player
player_width = 50
player_height = 30
player_x = SCREEN_WIDTH // 2 - player_width // 2
player_y = SCREEN_HEIGHT - player_height - 10
player_speed = 5

# Bullet
bullet_width = 5
bullet_height = 10
bullet_speed = 7
player_bullets = []
enemy_bullets = []

# Enemies
enemy_width = 40
enemy_height = 30
enemy_speed = 1
enemy_drop = 30
enemies = []
num_enemies = 5 * 11  # 5 rows, 11 columns
enemy_direction = 1  # 1 for right, -1 for left

def _create_enemies(self):
    self.enemies.clear()
    for row in range(5):
        for col in range(11):
            enemy_x = col * (enemy_width + 10) + 50
            enemy_y = row * (enemy_height + 10) + 50
            self.enemies.append([enemy_x, enemy_y])

# Score
score = 0
font = pygame.font.Font(None, 36)

# Game loop control
clock = pygame.time.Clock()
running = True

def draw_player():
    pygame.draw.rect(screen, GREEN, (player_x, player_y, player_width, player_height))

def draw_enemies():
    for enemy in enemies:
        pygame.draw.rect(screen, RED, (enemy[0], enemy[1], enemy_width, enemy_height))

def draw_bullets():
    for bullet in player_bullets:
        pygame.draw.rect(screen, WHITE, (bullet[0], bullet[1], bullet_width, bullet_height))
    for bullet in enemy_bullets:
        pygame.draw.rect(screen, BLUE, (bullet[0], bullet[1], bullet_width, bullet_height))

def move_player():
    global player_x
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x < SCREEN_WIDTH - player_width:
        player_x += player_speed

def shoot_bullet():
    if pygame.key.get_pressed()[pygame.K_SPACE]:
        player_bullets.append([player_x + player_width // 2 - bullet_width // 2, player_y])

def _move_bullets(self):
    for bullet in self.player_bullets[:]:
        bullet[1] -= bullet_speed
        if bullet[1] < 0:
            self.player_bullets.remove(bullet)
    for bullet in self.enemy_bullets[:]:
        bullet[1] += bullet_speed
        if bullet[1] > SCREEN_HEIGHT:
            self.enemy_bullets.remove(bullet)

def _enemy_shoot(self):
    if random.random() < 0.02 and self.enemies:  # Random chance for enemies to shoot
            enemy = random.choice(self.enemies)
            self.enemy_bullets.append([enemy[0] + enemy_width // 2 - bullet_width // 2, enemy[1] + enemy_height])

def _move_enemies(self):
    move_down = False  
    for enemy in self.enemies:
        enemy[0] += enemy_speed * self.enemy_direction
        if enemy[0] <= 0 or enemy[0] >= SCREEN_WIDTH - enemy_width:
            move_down = True
    if move_down:
        self.enemy_direction *= -1
        for enemy in self.enemies:
            enemy[1] += enemy_drop
            # 增加难度随着敌人下降而增加
            if enemy[1] > SCREEN_HEIGHT * 0.7:
                enemy_speed = min(3, enemy_speed + 0.01)

def _check_collisions(self):
    for p_bullet in player_bullets[:]:
        for enemy in enemies[:]:
            if (enemy[0] < p_bullet[0] < enemy[0] + enemy_width and
                enemy[1] < p_bullet[1] < enemy[1] + enemy_height):
                player_bullets.remove(p_bullet)
                enemies.remove(enemy)
                score += 10
                break
    for e_bullet in enemy_bullets[:]:
        if (player_x < e_bullet[0] < player_x + player_width and
            player_y < e_bullet[1] < player_y + player_height):
            return True  # Game over
    for enemy in enemies:
        if enemy[1] + enemy_height >= player_y:
            return True  # Enemies reached bottom
    return False

def draw_score():
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

# Main game loop for manual play
if __name__ == "__main__":
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        move_player()
        shoot_bullet()
        _move_bullets()
        _enemy_shoot()
        _move_enemies()

        game_over = _check_collisions()

        draw_player()
        draw_enemies()
        draw_bullets()
        draw_score()

        if game_over or not enemies:
            running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()