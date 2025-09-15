import gymnasium as gym
import numpy as np
import pygame
import random
import cv2

# 屏幕尺寸
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 玩家参数
player_width = 50
player_height = 30
player_speed = 5

# 子弹参数
bullet_width = 5
bullet_height = 10
bullet_speed = 7

# 敌人参数
enemy_width = 40
enemy_height = 30
enemy_speed = 1
enemy_drop = 30

class SpaceInvadersEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)  # 0: no-op, 1: left, 2: right, 3: shoot
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        # 游戏状态
        self.player_x = SCREEN_WIDTH // 2 - player_width // 2
        self.player_y = SCREEN_HEIGHT - player_height - 10
        self.enemies = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.score = 0
        self.enemy_direction = 1
        self.current_step = 0
        
        # Pygame初始化
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Space Invaders RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_x = SCREEN_WIDTH // 2 - player_width // 2
        self._create_enemies()
        self.player_bullets = []
        self.enemy_bullets = []
        self.score = 0
        self.enemy_direction = 1
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        done = False
        truncated = False

        # 执行动作
        if action == 1:  # 左移
            self.player_x = max(0, self.player_x - player_speed)
        elif action == 2:  # 右移
            self.player_x = min(SCREEN_WIDTH - player_width, self.player_x + player_speed)
        elif action == 3:  # 射击
            if len(self.player_bullets) < 5:
                self.player_bullets.append([self.player_x + player_width // 2 - bullet_width // 2, self.player_y])
                reward -= 2  # 射击有小惩罚

        # 游戏逻辑更新
        self._move_bullets()
        self._enemy_shoot()
        self._move_enemies()

        # 检查碰撞
        prev_enemy_count = len(self.enemies)
        collision = self._check_collisions()
        
        # 奖励计算
        enemies_destroyed = prev_enemy_count - len(self.enemies)
        if enemies_destroyed > 0:
            reward += 10 * enemies_destroyed
        
        reward -= 0.05  # 存活奖励
        
        # 游戏结束条件
        if collision:
            reward -= 10
            done = True
        elif not self.enemies:
            reward += 50
            done = True

        self.current_step += 1
        if self.current_step > 10000:
            truncated = True

        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def _get_obs(self):
        # 渲染到屏幕
        self.screen.fill(BLACK)
        
        # 绘制玩家
        pygame.draw.rect(self.screen, GREEN, (self.player_x, self.player_y, player_width, player_height))
        
        # 绘制敌人
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, RED, (enemy[0], enemy[1], enemy_width, enemy_height))
        
        # 绘制子弹
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, WHITE, (bullet[0], bullet[1], bullet_width, bullet_height))
        for bullet in self.enemy_bullets:
            pygame.draw.rect(self.screen, BLUE, (bullet[0], bullet[1], bullet_width, bullet_height))
        
        # 绘制分数
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # 获取屏幕数据
        obs_surface = pygame.surfarray.array3d(self.screen)
        obs_surface = obs_surface.swapaxes(0, 1)
        
        # 调整大小，但不进行归一化，并确保数据类型为uint8
        obs_surface = cv2.resize(obs_surface, (84, 84))
        # 注意：cv2.resize后可能返回float，如果需要则转换为uint8
        # 因为你的原始像素值是0-255的整数，所以通常resize后可以直接作为uint8处理
        if obs_surface.dtype != np.uint8:
            obs_surface = obs_surface.astype(np.uint8)
        
        return obs_surface # 直接返回uint8数组，范围[0, 255]

    def _create_enemies(self):
        self.enemies.clear()
        for row in range(5):
            for col in range(11):
                enemy_x = col * (enemy_width + 10) + 50
                enemy_y = row * (enemy_height + 10) + 50
                self.enemies.append([enemy_x, enemy_y])

    def _move_bullets(self):
        # 玩家子弹
        for bullet in self.player_bullets[:]:
            bullet[1] -= bullet_speed
            if bullet[1] < 0:
                self.player_bullets.remove(bullet)
        
        # 敌人子弹
        for bullet in self.enemy_bullets[:]:
            bullet[1] += bullet_speed
            if bullet[1] > SCREEN_HEIGHT:
                self.enemy_bullets.remove(bullet)

    def _enemy_shoot(self):
        if random.random() < 0.02 and self.enemies:
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

    def _check_collisions(self):
        # 玩家子弹与敌人碰撞
        for p_bullet in self.player_bullets[:]:
            for enemy in self.enemies[:]:
                if (enemy[0] < p_bullet[0] < enemy[0] + enemy_width and
                    enemy[1] < p_bullet[1] < enemy[1] + enemy_height):
                    self.player_bullets.remove(p_bullet)
                    self.enemies.remove(enemy)
                    self.score += 10
                    break
        
        # 敌人子弹与玩家碰撞
        for e_bullet in self.enemy_bullets[:]:
            if (self.player_x < e_bullet[0] < self.player_x + player_width and
                self.player_y < e_bullet[1] < self.player_y + player_height):
                return True  # 游戏结束
        
        # 敌人到达底部
        for enemy in self.enemies:
            if enemy[1] + enemy_height >= self.player_y:
                return True  # 游戏结束
        
        return False

    def render(self):
        # 渲染到屏幕
        self.screen.fill(BLACK)
        
        # 绘制玩家
        pygame.draw.rect(self.screen, GREEN, (self.player_x, self.player_y, player_width, player_height))
        
        # 绘制敌人
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, RED, (enemy[0], enemy[1], enemy_width, enemy_height))
        
        # 绘制子弹
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, WHITE, (bullet[0], bullet[1], bullet_width, bullet_height))
        for bullet in self.enemy_bullets:
            pygame.draw.rect(self.screen, BLUE, (bullet[0], bullet[1], bullet_width, bullet_height))
        

    def close(self):
        pygame.quit()