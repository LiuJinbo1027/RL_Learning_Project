# train.py
# This is the training layer that uses PPO to train on the environment.
# Requires stable-baselines3 and torch (install via pip if needed: pip install stable-baselines3 torch).

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from space_invaders_env import SpaceInvadersEnv

env = make_vec_env(SpaceInvadersEnv, n_envs=32)

model = PPO(
    ActorCriticCnnPolicy,
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./space_invaders_tensorboard/"
)

model.learn(total_timesteps= 20000 * 100)

model.save("ppo_space_invaders_cnn")