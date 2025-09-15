import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 超参数
GRID_SIZE = 12  # 网格大小
BLOCK_SIZE = 20  # 每个块像素大小
FPS = 30  # 渲染帧率（训练时可调慢）
EPISODES = 40000  # 训练episode数
BATCH_SIZE = 64
GAMMA = 0.99  # 折扣因子
LEARNING_RATE = 0.0001
FURTHER_SHAPED_REWARD_SCALE = 0.1 # 远离距离差奖励缩放因子
CLOSER_SHAPED_REWARD_SCALE = 0.1 # 靠近距离差奖励缩放因子
RENDER_EVERY = 40000 # 每隔多少次渲染一局游戏

# PPO 专有超参数
CLIP_EPS = 0.2  # 策略剪裁范围
PPO_EPOCHS = 4  # 每个轨迹的更新 epoch 数
VALUE_COEF = 0.5  # Critic 损失权重
ENTROPY_COEF = 0.01  # Entropy 正则化权重
MAX_GRAD_NORM = 0.5  # 梯度裁剪规范
LAMBDA = 0.95  # GAE 的 lambda 参数

MODEL_PATH = "first_4w_snake_ppo.pth"

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
        
        # 启发式奖励：计算当前距食物距离
        head = self.snake[0]
        old_dist = abs(head[0] -self.food[0]) + abs(head[1] - self.food[1])
        
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
        base_reward = 0
        if head == self.food:
            self.score += 1
            self.food =self._place_food()
            base_reward = 30
        else:
            self.snake.pop()

        # 计算步后距离
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        # 计算启发式奖励
        delta_dist = old_dist - new_dist # 靠近为正，远离为负
        shaped_reward = CLOSER_SHAPED_REWARD_SCALE * delta_dist if delta_dist > 0 else FURTHER_SHAPED_REWARD_SCALE * delta_dist

        reward = base_reward + shaped_reward

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

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        # 共享卷积骨干
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * GRID_SIZE * GRID_SIZE, 128)
        # Actor: 输出动作 logits
        self.actor = nn.Linear(128, 4)
        # Critic: 输出状态价值 V(s)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x): # 向前传播
        # 第一个卷积层处理输入，应用ReLU激活函数，提取低级特征
        x = F.relu(self.conv1(x))
        # 通过第二个卷积层，进一步提取高级特征
        x = F.relu(self.conv2(x))
        # 将特征图展开为1d向量
        x = x.view(x.size(0), -1)
        # 通过全连接层（fc）将展平向量映射到128维隐藏表示，并应用ReLU激活
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x) # 输出策略，价值

def compute_gae(next_value, rewards, masks, values, gamma, lam):
    """
    计算 Generalized Advantage Estimation (GAE 优势估计) 和 returns
    GAE是优势函数的一种平滑估计方法，能减少方差，提高训练稳定性
    next_value：下一个状态的价值，用于引导
    rewards：奖励列表
    masks：掩码列表，done时为0，否则为1，用于处理episode结束
    values：当前价值估计列表
    gamma：折扣因子
    lam：lambda参数，用于控制偏差-方差权衡
    """
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))): # 从后向前遍历奖励列表，避免未来信息泄露
        # 误差：衡量当前奖励加上折扣未来价值与当前价值估计的误差
        delta = rewards[step] + gamma * next_value * masks[step] - values[step] 
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae) # 反转优势序列，确保正向输出
        next_value = values[step] # 用于下一次迭代
    return returns, advantages

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device:{device}")
    env = SnakeEnv() # 创建贪吃蛇环境
    ppo = PPO().to(device) # PPO 网络
    optimizer = optim.Adam(ppo.parameters(), lr=LEARNING_RATE) # 优化器

    # 收集数据以绘制曲线
    scores = [] # 每个episode的分数
    rewards = [] # 每个episode的累积奖励

    """=== 主训练循环 ==="""
    for episode in range(EPISODES): # 按指定次数进行训练
        state = env.reset() # 重置环境
        states_list, actions_list, old_log_probs_list, values_list, rewards_list, dones_list = [], [], [], [], [], []
        episode_reward = 0
        done = False

        do_render = (episode % RENDER_EVERY == 0) # 判断本局是否渲染
        
        while not done: # 直到游戏结束
            # 使用策略采样动作（stochastic for exploration）
            with torch.no_grad():
                state_t = torch.tensor(state).unsqueeze(0).float().to(device)
                logits, value = ppo(state_t)
                probs = F.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
                log_prob = action_dist.log_prob(torch.tensor(action).to(device))
            
            next_state, reward, done = env.step(action) # 执行动作
            episode_reward += reward 
            
            # 收集轨迹数据
            states_list.append(state)
            actions_list.append(action)
            old_log_probs_list.append(log_prob.item())
            values_list.append(value.item())
            rewards_list.append(reward)
            dones_list.append(done)
            
            state = next_state
            if do_render:
                env.render() # 实时渲染

        # 计算下一个状态的价值（用于 GAE bootstrap）
        with torch.no_grad():
            next_state_t = torch.tensor(state).unsqueeze(0).float().to(device)
            _, next_value = ppo(next_state_t)
            next_value = next_value.item()

        # 计算 returns 和 advantages
        masks = [0 if d else 1 for d in dones_list]  # mask = 1 - done (done时为0)
        returns, advantages = compute_gae(next_value, rewards_list, masks, values_list, GAMMA, LAMBDA)

        # 转换为张量
        states_t = torch.tensor(np.array(states_list)).float().to(device)
        actions_t = torch.tensor(actions_list).long().to(device)
        old_log_probs_t = torch.tensor(old_log_probs_list).float().to(device)
        returns_t = torch.tensor(returns).float().to(device)
        advantages_t = torch.tensor(advantages).float().to(device)

        # 归一化 advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO 更新（多 epoch）
        for _ in range(PPO_EPOCHS):
            logits, values = ppo(states_t)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = - (probs * log_probs).sum(-1).mean()

            # Actor loss (clipped surrogate)
            log_probs_act = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            ratios = torch.exp(log_probs_act - old_log_probs_t)
            surr1 = ratios * advantages_t
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t
            actor_loss = - torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(1), returns_t)

            # 总损失
            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

            # 更新
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        scores.append(env.score)
        rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}/{EPISODES}, Score: {env.score}, Reward: {episode_reward}")

    # 保存模型
    torch.save(ppo.state_dict(), "snake_ppo.pth")
    print("训练完成，模型保存为 snake_ppo.pth")

    # 新增：绘制学习曲线
    episodes_list = list(range(1, EPISODES + 1)) #x轴：1--episodes
    plt.figure(figsize=(12, 6)) # 画布尺寸宽12英寸高6英寸
    
    # 子图1: episode vs score
    plt.subplot(1, 2, 1) # 1行2列的第一个子图
    plt.plot(episodes_list, scores, label='Score') # x轴：episodes_list y轴：scores 标题：label
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score over Episodes')
    plt.legend()
    
    # 子图2: episode vs reward
    plt.subplot(1, 2, 2)
    plt.plot(episodes_list, rewards, label='Episode Reward', color='orange') # 加以颜色区分
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    plt.legend()
    
    plt.tight_layout() # 自动调节子图间距，防止重叠
    plt.show()  # 显示图像（可选：plt.savefig('learning_curves.png') 保存为文件）

"""========================================================================================="""

def demo(model_path = MODEL_PATH):
    """
    演示函数：加载训练好的模型，运行贪吃蛇游戏，渲染每一帧。
    model_path: 模型权重文件路径，默认为'snake_ppo.pth'。
    """
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Demo using device: {device}")

    # 初始化环境模型
    env = SnakeEnv()
    ppo = PPO().to(device)

    # 加载模型权重
    try:
        ppo.load_state_dict(torch.load(model_path))
        ppo.eval()
        print(f"Loading model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    
    # 运行演示
    state = env.reset()
    done = False
    episode_reward = 0
    score =0

    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state).unsqueeze(0).float().to(device)
            logits, _ = ppo(state_t) # 仅需actor
            probs = F.softmax(logits, dim = -1)
            action = torch.argmax(probs, dim = -1).item() # 选择概率最大的动作

        state, reward, done = env.step(action)
        episode_reward += reward
        if reward >= 30:  # 假设吃到食物奖励为30
            score += 1
        env.render()
        
        # 检查用户是否关闭窗口
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    print(f"Demo finished. Score: {score}, Total Reward: {episode_reward}")
    env.close()


"""========================================================================================="""

def continue_train(model_path = MODEL_PATH, additional_episodes =10000):
    """
    继续训练函数：加载模型，继续训练指定的额外episode数。
    model_path: 模型权重文件路径，默认为'snake_ppo.pth'。
    additional_episodes: 继续训练的episode数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Demo using device: 『{device}")

    # 初始化环境，模型
    env = SnakeEnv()
    ppo = PPO().to(device)
    optimizer = optim.Adam(ppo.parameters(), lr = LEARNING_RATE)

    # 加载模型权重
    try:
        ppo.load_state_dict(torch.load(model_path))
        print(f"Loading model from {model_path}")
    except Exception as e:
        print(f"Error loading model:{e}. Starting from scratch.")

    scores = []
    rewards = []

"""========================================================================================="""

if __name__ == "__main__":
    mode = input("Enter mode: (train/demo/continue)").strip().lower()
    if mode == "train":
        train()
    elif mode == "demo":
        demo()
    else:
        print("Invalid mode. Please choose 'train', 'demo', or 'continue'.")