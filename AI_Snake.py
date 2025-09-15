import gymnasium
from gymnasium import spaces
import numpy as np
import random
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 超参数
GRID_SIZE = 12  # 网格大小
BLOCK_SIZE = 20  # 每个块像素大小
FPS = 30  # 渲染帧率
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

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
        
        # Gymnasium 空间定义
        self.action_space = spaces.Discrete(4)  # 4 个离散动作
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
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
            return self._get_obs(), -1, True, False, {}  # 死亡惩罚 -1

        self.snake.insert(0, head)

        # 检查食物
        base_reward = 0
        if head == self.food:
            self.score += 1
            self.food = self._place_food()
            base_reward = 30
        else:
            self.snake.pop()

        # 计算新距离
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        delta_dist = old_dist - new_dist  # 靠近 >0，远离 <0

        # 奖惩系数 = 1 / 蛇长度
        snake_length = len(self.snake)
        coef = 1.0 / snake_length if snake_length > 0 else 0.1  # 避免除零

        # 塑造奖励：靠近加分，远离扣分
        shaped_reward = coef * delta_dist

        reward = base_reward + shaped_reward

        return self._get_obs(), reward, self.done, False, {}  # truncated=False

    def _get_obs(self):
        state = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[0][self.food] = 1.0  # 食物通道
        for i, pos in enumerate(self.snake):
            if i == 0:
                state[2][pos] = 1.0  # 头通道
            else:
                state[1][pos] = 1.0  # 身通道
        return state

    def render(self):
        self.screen.fill(BLACK)
        for i, pos in enumerate(self.snake):
            color = RED if i == 0 else GREEN
            pygame.draw.rect(self.screen, color, (pos[1] * BLOCK_SIZE, pos[0] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, WHITE, (self.food[1] * BLOCK_SIZE, self.food[0] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 128):
        super(CustomCnnExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 3 通道
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 计算展平后的维度
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def train_sb3():
    # 并行环境（加速训练，n_envs=4 表示 4 个并行环境）
    env = make_vec_env(SnakeEnv, n_envs=4)

    # 配置 TensorBoard 日志
    logger = configure("./tensorboard_logs/", ["stdout", "tensorboard"])
    
    # 创建 PPO 模型，使用 CnnPolicy
    policy_kwargs = dict(
    features_extractor_class=CustomCnnExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128]  # 共享的 FC 层
    )
    model = PPO(
        "CnnPolicy",  # SB3 内置 CNN 策略
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0001,
        n_steps=2048,  # 每轮收集步数
        batch_size=64,
        n_epochs=4,  # PPO 更新 epoch
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/",  # TensorBoard 日志路径
        device="cuda" if th.cuda.is_available() else "cpu"
    )
    model.set_logger(logger)

    # 训练
    model.learn(total_timesteps=40000 * 100)  # 约 40000 episode（假设每 episode 100 步）

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