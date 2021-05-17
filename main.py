from train import evaluate_policy
import gym
import numpy as np
import torch
from ddpg import DDPG

# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ ==  "__main__":
    # Initialize parameters
    env_name = "HalfCheetah-v2"
    seed = 0
    eval_episodes = 10

    # file name for pretrained model
    file_name = "%s_%s_%s" % ("DDPG", env_name,str(seed))

    env = gym.make(env_name) # define env

    # random seed
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # state/action dimentions and max actions
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # create agent and load policy
    agent = DDPG(state_dim, act_dim, max_action)
    agent.load(file_name, './models/')
    _ = evaluate_policy(agent, eval_episodes=eval_episodes)

    # main loop - render environment and agent inference
    for i_episodes in range(100):
        obs = env.reset()
        for t in range(200):
            env.render()

            action = agent.select_action(obs)
            # action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
