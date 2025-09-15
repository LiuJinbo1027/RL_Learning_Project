import gymnasium
from gymnasium import spaces
import numpy as np
import random
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 超参数
GRID_SIZE = 12  # 网格大小
BLOCK_SIZE = 20  # 每个块像素大小
OBS_SCALE = 4   # 观察缩放因子
FPS = 24  # 渲染帧率
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
FOOD_REWARD = 10
DIE_REWARD = -15

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class SnakeEnv(gymnasium.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        pygame.display.init()  # 只初始化显示模块，避免音频问题
        self.screen = pygame.display.set_mode((GRID_SIZE * BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE))
        pygame.display.set_caption("Snake RL with Gymnasium and SB3")
        self.clock = pygame.time.Clock()

        # 创建观察表面，放大观察尺寸
        self.obs_surface = pygame.Surface((GRID_SIZE * OBS_SCALE, GRID_SIZE * OBS_SCALE))
        self.action_space = spaces.Discrete(4)  # 4 个离散动作
        
        # 修改观察空间为放大后的尺寸
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(3, GRID_SIZE * OBS_SCALE, GRID_SIZE * OBS_SCALE), 
            dtype=np.uint8
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]  # 初始蛇身
        self.direction = random.choice(ACTIONS)  # 随机初始方向
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self.snake_directions = [self.direction] # 跟踪每个蛇身部分的方向
        return self._get_obs(), {}  # 返回观察和 info

    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # truncated=False

        head = self.snake[0]
        old_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])  # 曼哈顿距离

        # 更新方向（防止反向）
        new_dir = ACTIONS[action]
        if (new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]):
            new_dir = self.direction
        self.direction = new_dir

        # 移动蛇头
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # 检查碰撞
        if (head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE or head in self.snake):
            self.done = True
            return self._get_obs(), DIE_REWARD, True, False, {}  # 死亡惩罚

        self.snake.insert(0, head)
        self.snake_directions.insert(0, self.direction)

        # 检查食物
        base_reward = 0
        if head == self.food:
            self.score += 1
            self.food = self._place_food()
            base_reward = FOOD_REWARD
        else:
            self.snake.pop()
            self.snake_directions.pop()

        # 计算新距离
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        delta_dist = old_dist - new_dist  # 靠近 >0，远离 <0

        # 奖惩系数 = 1 / 蛇长度
        snake_length = len(self.snake)
        coef = 1.0 / snake_length if snake_length > 0 else 0.1  # 避免除零

        # 塑造奖励：靠近加分，远离扣分
        penalty_factor = 1.5 # 远离惩罚放大因子
        if delta_dist > 0:
            shaped_reward = coef * delta_dist
        else:
            shaped_reward = coef * delta_dist * penalty_factor

        reward = base_reward + shaped_reward

        return self._get_obs(), reward, self.done, False, {}  # truncated=False

    def _get_obs(self):
        # 清空观察表面
        self.obs_surface.fill(BLACK)
        
        # 绘制食物 (白色)
        pygame.draw.rect(
            self.obs_surface, 
            WHITE, 
            (self.food[1] * OBS_SCALE, self.food[0] * OBS_SCALE, OBS_SCALE, OBS_SCALE)
        )
        
        # 绘制蛇身，使用渐变效果
        snake_length = len(self.snake)
        for i, pos in enumerate(self.snake):
            # 计算渐变颜色：蛇头最亮，越往尾部越暗
            intensity = int(255 * (1 - i / (snake_length * 1.5)))  # 确保不会变太暗
            intensity = max(50, intensity)  # 确保最低亮度
            
            if i == 0:  # 蛇头
                color = (255, 0, 0)  # 红色
            else:  # 蛇身
                color = (0, intensity, 0)  # 绿色渐变
            
            pygame.draw.rect(
                self.obs_surface, 
                color, 
                (pos[1] * OBS_SCALE, pos[0] * OBS_SCALE, OBS_SCALE, OBS_SCALE)
            )
        
        # 将Pygame表面转换为numpy数组
        obs_array = pygame.surfarray.array3d(self.obs_surface)
        # 调整轴顺序以适应PyTorch (H, W, C) -> (C, H, W)
        obs_array = obs_array.transpose(2, 0, 1)
        
        return obs_array

    def render(self):
        # 清屏
        self.screen.fill(BLACK)
        # 绘制食物
        pygame.draw.rect(
            self.screen, 
            WHITE, 
            (self.food[1] * BLOCK_SIZE, self.food[0] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        )
        
        # 绘制蛇身，使用渐变效果
        snake_length = len(self.snake)
        for i, pos in enumerate(self.snake):
            # 计算渐变颜色：蛇头最亮，越往尾部越暗
            intensity = int(255 * (1 - i / (snake_length * 1.5)))  # 确保不会变太暗
            intensity = max(50, intensity)  # 确保最低亮度
            
            if i == 0:  # 蛇头
                color = RED
            else:  # 蛇身
                color = (0, intensity, 0)  # 绿色渐变
            
            pygame.draw.rect(
                self.screen, 
                color, 
                (pos[1] * BLOCK_SIZE, pos[0] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            )
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

class EnhancedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 计算输入尺寸 (GRID_SIZE * OBS_SCALE)
        input_size = GRID_SIZE * OBS_SCALE
        
        # 计算卷积后的尺寸
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        conv1w = conv2d_size_out(input_size, 4, 2)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        conv2w = conv2d_size_out(conv1w, 3, 1)
        
        # 第三层卷积
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv3w = conv2d_size_out(conv2w, 3, 1)
        
        # 第四层卷积
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        conv4w = conv2d_size_out(conv3w, 3, 1)
        
        # 计算全连接层输入尺寸
        linear_input_size = conv4w * conv4w * 128
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)

def train_sb3():
    # 创建环境
    env = DummyVecEnv([lambda: SnakeEnv() for _ in range(16)])  # 减少并行环境数量
    
    # 配置 TensorBoard 日志
    logger = configure("./tensorboard_logs/", ["stdout", "tensorboard"])
    
    # 使用自定义的CNN策略
    policy_kwargs = dict(
        features_extractor_class=EnhancedCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]))
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/",
        device="cuda" if th.cuda.is_available() else "cpu"
    )
    model.set_logger(logger)

    # 训练
    model.learn(total_timesteps=80000 * 100)

    # 保存模型
    model.save("snake_ppo_sb3")

    print("训练完成，模型保存为 snake_ppo_sb3.zip")
    print("使用 TensorBoard 可视化：tensorboard --logdir ./tensorboard_logs/")

def demo_sb3(model_path="snake_ppo_sb3"):
    env = SnakeEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # 贪婪策略
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.close()
    print("演示结束")

if __name__ == "__main__":
    mode = input("Enter mode (train/demo): ").strip().lower()
    if mode == "train":
        train_sb3()
    elif mode == "demo":
        demo_sb3()
    else:
        print("Invalid mode. Please choose 'train' or 'demo'.")