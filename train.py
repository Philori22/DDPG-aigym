from replaybuffer import ReplayBuffer
from ddpg import DDPG
import gym
import torch
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

env_name = "HalfCheetah-v2"
# environment
env = gym.make(env_name)
# env.render()

 # parameters
seed = 0 # Random seed number
start_timesteps = 1e4 # numbe rof iterations before which the model randomly choses an action, and after which it starts to use the plicy netowk
eval_freq = 5e3
max_timesteps = 15e5
save_models = True
expl_noise = 0.1
batch_size = 100
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2


# Selecting the device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print ("---------------------------------------")

    return avg_reward

def main():

  # filename for saved models
  file_name = "%s_%s_%s" % ("DDPG", env_name, str(seed))
  print("------------------------")
  print("Settings: %s" % (file_name))
  print("---------------------------")

  # set seeds and information on states and actions

  env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  max_action = float(env.action_space.high[0])

  # create agent
  policy = DDPG(state_dim, action_dim, max_action)
  replay_buffer = ReplayBuffer()
  evaluations = [evaluate_policy(policy)]

  # initialize variables
  total_timesteps = 0
  timesteps_since_eval = 0
  episode_num = 0
  done = True
  t0 = time.time()

  # writer
  writer = SummaryWriter("runs/%s_%s_%s" % ("DDPG", env_name, str(seed)))

  # We start the main loop over 500,000 timesteps
  while total_timesteps < max_timesteps:

    # if the episode is done
    if done:

      # If we are not at the very beginning, we start the training process of the model
      if total_timesteps != 0:
        print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
        policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

      # we evaluate the episode and  we save the policy
      if timesteps_since_eval >= eval_freq:
        timesteps_since_eval %= eval_freq
        evaluations.append(evaluate_policy(policy))
        policy.save(file_name, directory="./models")
        np.save("./results/%s" % (file_name), evaluations)
        

      # when the training step is done, we reset the state of the environment
      obs = env.reset()

      # Set the Done to False
      done = False

      # Set rewards and episode timesteps to zero
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1

    # Before 10000 timesteps, we play random actions
    if total_timesteps < start_timesteps:
      action = env.action_space.sample()
    else:   # After 10000 timesteps, we switch to the model
      action = policy.select_action(np.array(obs))
      # If the explore_noise parameter is not 0, we add noise to the action and we clip it
      if expl_noise != 0:
        action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
      
    # The agent performs the action in the environment, then reaches the next state and receives the reward
    new_obs, reward, done, _ = env.step(action)

    # we check if the episode is done
    done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

    # we increase the total reward
    episode_reward += reward

    # we store the new transition into Experience Replay memory (ReplayBuffer)
    replay_buffer.add((obs,new_obs,action,reward, done_bool))

    # we update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1


    # Cumulative Reward - Tensor Board
    writer.add_scalar("The average episodic reward per epoch", episode_reward, episode_num)


  # We add the last policy evaluation to our list of evaluations and we save our model
  evaluations.append(evaluate_policy(policy))
  if save_models: policy.save("%s" % (file_name), directory="./models")
  np.save("./results/%s" % (file_name), evaluations)

  writer.close()




if __name__ == '__main__':
    main()
    experiment = ''