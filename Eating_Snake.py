import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time

# 超参数
GRID_SIZE = 10  # 网格大小
BLOCK_SIZE = 20  # 每个块像素大小
FPS = 30  # 渲染帧率（训练时可调慢）
EPISODES = 1000  # 训练episode数
BATCH_SIZE = 64
GAMMA = 0.99  # 折扣因子
EPS_START = 1.0  # 初始探索率
EPS_END = 0.05  # 最小探索率
EPS_DECAY = 0.995  # 探索衰减
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE = 100  # 目标网络更新频率

# 动作定义：0上，1下，2左，3右
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((GRID_SIZE*BLOCK_SIZE, GRID_SIZE*BLOCK_SIZE))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)] # 初始蛇身
        self.direction = random.choice(ACTIONS) # 随机初始方向
        self.food = self._place_food()
        self.score = 0
        self.done = False
        return self._get_state()
    
    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if food not in self.snake:
                return food
    
    def step(self, action):
        if self.done:
            return self._get_state(), 0, True
        
        # 更新方向（防止反向）：
        new_dir = ACTIONS[action]
        if(new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]):
            new_dir = self.direction # 保持原方向
        self.direction = new_dir

        # 移动蛇头
        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # 检查碰撞
        if(head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE or head in self.snake):
            self.done = True
            return self._get_state(), -1, True # 死亡惩罚
        
        self.snake.insert(0, head)

        # 检查食物
        reward = 0
        if head == self.food:
            self.score += 1
            self.food =self._place_food()
            reward = 1
        else:
            self.snake.pop()
        
        return self._get_state(), reward, self.done
    
    def _get_state(self):
        # 状态：3通道网络（蛇身=1， 蛇头=2， 食物=3）
        state = np.zeros((3, GRID_SIZE, GRID_SIZE))
        state[0][self.food] = 1 # 食物通道
        for i, pos in enumerate(self.snake):
            if i == 0:
                state[2][pos] = 1 # 头通道
            else:
                state[1][pos] = 1 # 身通道
        return state # 返回（3， H， W）张量
        
    def render(self):
        self.screen.fill(BLACK)
        # 绘制蛇
        for i, pos in enumerate(self.snake):
            color = RED if i == 0 else GREEN # 头红色，身绿色
            pygame.draw.rect(self.screen, color, (pos[1]*BLOCK_SIZE, pos[0]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # 绘制食物
        pygame.draw.rect(self.screen, WHITE, (self.food[1]*BLOCK_SIZE, self.food[0]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones =zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
    
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 卷积层1:输入3通道（食物，蛇身，蛇头），输出16通道特征图
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 卷积层2:输入16通道，输出32通道特征图
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 全连接层1:将展平的特征向量（32*10*10=3200维）映射到128维
        self.fc1 = nn.Linear(32* GRID_SIZE* GRID_SIZE, 128)
        # 输出层：映射到四个动作的Q值（上下左右）
        self.fc2 = nn.Linear(128, 4)
        
    # 向前传播
    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def train():
    env = SnakeEnv() # 创建贪吃蛇环境
    policy_net = DQN() # 策略网络（实时更新）
    target_net = DQN() # 目标网络（定期同步）
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE) # 优化器
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE) # 经验回放池
    
    eps = EPS_START # 初始探索率
    steps = 0

    """=== 主训练循环 ==="""
    for episode in range(EPISODES): # 按指定次数进行训练
        state = env.reset() # 重制环境
        episode_reward = 0
        done = False
        
        while not done: # 直到游戏结束
            # epsilon-greedy动作选择
            if random.random() < eps: # 探索
                action = random.randint(0,3)
            else:
                with torch.no_grad(): # 禁用梯度
                    q_values = policy_net(torch.tensor(state).unsqueeze(0).float())
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action) # 执行动作
            episode_reward += reward 
            replay_buffer.push(state, action, reward, next_state, done) # 存储经验
            state = next_state

            # 更新网络
            if len(replay_buffer) >= BATCH_SIZE: # 当经验池充足时
                # 从回放池采样
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                # 转换为pytorch向量
                states = torch.tensor(states).float()
                actions = torch.tensor(actions).long().unsqueeze(1)
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()

                # 计算当前Q值(policy网络)
                q_values = policy_net(states).gather(1, actions).squeeze(1)
                # 计算目标Q值（target网络）
                with torch.no_grad(): # 禁用梯度
                    next_q_values = target_net(next_states).max(1)[0]
                    expected_q = rewards + GAMMA * next_q_values * (1 - dones)
                # 计算损失并反向传播
                loss = F.mse_loss(q_values, expected_q)
                optimizer.zero_grad() # 梯度清零
                loss.backward() # 反向传播
                optimizer.step() # 参数更新

            env.render() # 实时渲染
            steps += 1

            # 更新目标网络
            if steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
        eps = max(EPS_END, eps * EPS_DECAY)
        print(f"Episode {episode + 1}/{EPISODES}, Score: {env.score}, Reward: {episode_reward}, Eps: {eps:.2f}")

    # 保存模型
    torch.save(policy_net.state_dict(), "snake_dqn.pth")
    print("训练完成，模型保存为 snake_dqn.pth")

if __name__ == "__main__":
    train()