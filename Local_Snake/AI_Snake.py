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
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 超参数
GRID_SIZE = 12  # 网格大小
BLOCK_SIZE = 20  # 每个块像素大小
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

        self.obs_surface = pygame.Surface((GRID_SIZE, GRID_SIZE))  # 用于观察的表面
        self.action_space = spaces.Discrete(4)  # 4 个离散动作
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
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
            base_reward = FOOD_REWARD * len(self.snake)  # 食物奖励与蛇长成正比
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
            (self.food[1], self.food[0], 1, 1)
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
                (pos[1], pos[0], 1, 1)
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

# 自定义小型CNN特征提取器，适用于小尺寸输入
class SmallCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 128):
        super(SmallCNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 3 通道 (RGB)
        
        # 使用较小的卷积核和步长以适应小尺寸输入
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算展平后的维度
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def train_sb3():
    # 创建环境
    env = make_vec_env(SnakeEnv, n_envs=32)

    # 配置 TensorBoard 日志
    logger = configure("./tensorboard_logs/", ["stdout", "tensorboard"])
    
    # 使用自定义的小型CNN策略
    policy_kwargs = dict(
        features_extractor_class=SmallCNNExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[64, 64]  # 较小的全连接层
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,  # 较小的步数
        batch_size=64,  # 较小的批次大小
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/",
        device="cuda" if th.cuda.is_available() else "cpu"
    )
    model.set_logger(logger)

    # 训练
    model.learn(total_timesteps=130000 * 100)

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