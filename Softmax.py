import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    """ 多臂老虎机环境 """
    def __init__(self, arm_probs):
        self.arm_probs = arm_probs # 每个臂的奖励概率
        self.num_arms = len(arm_probs)

    def pull(self, arm):
        """ 拉动指定的臂，返回奖励 """
        if np.random.random() < self.arm_probs[arm]:
            return 1 # 获得奖励
        return 0 # 没获得奖励

class SoftmaxAgent:
    """ Softmax Agent with Policy Gradient Update """
    def __init__(self, num_arms, temperature=0.1, alpha=0.1):
        self.num_arms = num_arms
        self.temperature = temperature
        self.alpha = alpha # 梯度策略的学习率
        self.H = np.zeros(num_arms) # 偏好值
    
    def select_action(self):
        """ Select action based on Softmax probabilities """
        exp_prefs = np.exp(self.H / self.temperature)
        probabilities = exp_prefs / np.sum(exp_prefs)
        action = np.random.choice(self.num_arms, p=probabilities)
        return action, probabilities  # Return action and current pi for update
    
    def update(self, action, reward, probabilities):
        """ Update preferences using policy gradient: H(a) += alpha * R * (I(a==A) - pi(a)) """
        for a in range(self.num_arms):
            indicator = 1 if a == action else 0
            self.H[a] += self.alpha * reward * (indicator - probabilities[a])


def run_experiment(num_arms, temperature, num_steps = 1000, alpha = 0.1):
    """ 运行实验 """
    # 创建环境（真实概率分布）
    true_probs = np.array([0.7, 0.3, 0.5, 0.1, 0.9])
    env = BanditEnvironment(true_probs)
    
    # 创建智能体
    agent = SoftmaxAgent(num_arms, temperature, alpha)

    # 记录结果
    rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps)
    optimal_arm = np.argmax(true_probs)

    for step in range(num_steps):
        # 选择动作
        action, probabilities = agent.select_action()
        # 执行动作并获得奖励
        reward = env.pull(action)
        # 更新智能体
        agent.update(action, reward, probabilities)
        # 记录结果
        rewards[step] = reward
        optimal_actions[step] = 1 if action == optimal_arm else 0
    
    final_preferences = agent.H
    return rewards, optimal_actions, final_preferences
    
# 运行实验
num_arms = 5
num_steps = 1000
temperature = 0.2 # 温度参数：值越大探索越多，值越小利用越多
alpha = 0.1

rewards, optimal_actions, final_preferences = run_experiment(
    num_arms, temperature, num_steps, alpha
)

# 计算累计奖励和最优动作比例
cumulative_rewards = np.cumsum(rewards)
optimal_ratio = np.cumsum(optimal_actions) / (np.arange(num_steps) + 1)

# 可视化结果
plt.figure(figsize=(15, 10))

# 累积奖励图
plt.subplot(2, 2, 1)
plt.plot(cumulative_rewards)
plt.xlabel('Step') # 步数
plt.ylabel('Cumulative Reward') # 累计奖励
plt.title('Cumulative Reward Curve') # 累积奖励曲线
plt.grid(True)

# 最优动作比例图
plt.subplot(2, 2, 2)
plt.plot(optimal_ratio)
plt.xlabel('Step') # 步数
plt.ylabel('Optimal Arm Selection Ratio') # 选择最优臂的比例
plt.title('Policy Improvement Process') # 策略优化过程
plt.ylim([0, 1.1])
plt.grid(True)

# 最终偏好值
plt.subplot(2, 2, 3)
plt.bar(range(num_arms), final_preferences)
plt.xlabel('Policy Improvement Process') # 策略优化过程
plt.ylabel('Preference Value') # 偏好值
plt.title('Final Preference Value') # 最终偏好值分布
plt.grid(True)

# 真实概率与偏好值对比
plt.subplot(2, 2, 4)
plt.bar(range(num_arms), [0.7, 0.3, 0.5, 0.1, 0.9], alpha=0.7, label='True Probabilities') # 真实概率
plt.plot(range(num_arms), final_preferences / np.max(final_preferences), 
         'ro-', label='Normalized Preferences') # 偏好值(归一化)
plt.xlabel('Arm') # 臂
plt.ylabel('Value') # 值
plt.title('True Probabilities vs. Learned Preferences') # 真实概率 vs. 学习到的偏好值
plt.legend()
plt.grid(True)

plt.tight_layout()

# 打印最终策略
exp_prefs = np.exp(final_preferences / temperature)
probabilities = exp_prefs / np.sum(exp_prefs)
print("\n最终策略(选择每个臂的概率):")
for i, prob in enumerate(probabilities):
    print(f"臂 {i}: {prob:.4f}")

plt.show()
