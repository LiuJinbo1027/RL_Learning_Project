# play.py
# This script demonstrates the trained AI playing the game.
# It loads the PPO model and runs it in the environment with rendering.

import pygame
from stable_baselines3 import PPO
from space_invaders_env import SpaceInvadersEnv

# Initialize Pygame for rendering
pygame.init()

# Create the environment (single instance, not vectorized)
env = SpaceInvadersEnv()

# Load the trained model
model = PPO.load("ppo_space_invaders_cnn")

# Demonstration function
def demonstrate_ai(num_episodes=5):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()  # Render the game screen
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# Run the demonstration
if __name__ == "__main__":
    demonstrate_ai()
    env.close()