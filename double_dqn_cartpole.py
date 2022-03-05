import gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from IPython.display import clear_output


# Hyperparameters
EPISODES     = 200    # number of episodes
EPS_START    = 0.9    # epsilon-greedy threshold start value
EPS_END      = 0.05   # epsilon-greedy threshold end value
EPS_DECAY    = 200    # epsilon-greedy threshold decay
GAMMA        = 0.75   # Q-learning discount factor
LEARN_RATE   = 0.001  # optimizer learning rate
HIDDEN_LAYER = 164    # neural network hidden layer size
BATCH_SIZE   = 64     # Q-learning batch size

# Run with GPU
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor  = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor      = FloatTensor

# ----------------------------- Classes ----------------------------- #
# DQN Class
class DQN(nn.Module):
    def __init__(self):
        # nn.Module.__init__(self)
        super(DQN, self).__init__()
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

# ReplayMemory Class to sample from memory/experience
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
# --------------------------- End Classes --------------------------- #


# Initialize CartPole Model
env = gym.make('CartPole-v0')

# Instantiate model and target networks
model  = DQN()
target = DQN()
if use_cuda:
    model.cuda()
    target.cuda()

# Instantiate replay memory
memory = ReplayMemory(10000)

# Loss Function
criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), LEARN_RATE)


# ---------------------------- Functions ---------------------------- #
def select_action(state, train=True):
    """This function chooses an action to take based on the state: move cart left or right"""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if train:
        if sample > eps_threshold:
            return model((state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(2)]])
    else:
        return model((state).type(FloatTensor)).data.max(1)[1].view(1, 1)

def run_episode(episode, env, rewards_list):
    """This function executes an episode for the CartPole game"""
    # Update the weights of the target network after n steps
    n = 5
    if episode % n == 0:
        target.load_state_dict(model.state_dict())

    state = env.reset()
    done = False
    rewards = 0
    while not done:
        # Select action and step to next state
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = env.step(action[0, 0].item())

        # Update rewards
        rewards += reward

        if done:
            reward = -1

        # Save state, action, next_state and reward to memory for experience replay
        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        # Train model
        train()

        # Update state
        state = next_state

    rewards_list.append(rewards)
    return rewards_list

def train():
    """This function executes the training phase for the DQN agent"""
    if len(memory) < BATCH_SIZE:
        return

    # Random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state      = torch.cat(batch_state)
    batch_action     = torch.cat(batch_action)
    batch_reward     = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    # Zero-out the gradients
    optimizer.zero_grad()

    # Compute neural network predictions (forward pass)
    # current Q values are estimated by model neural network for all actions
    current_q_values  = model(batch_state).gather(1, batch_action)

    # expected Q values are estimated by target neural network, which gives maximum Q value
    max_next_q_values = target(batch_next_state).detach().max(1)[0]
    # ground truth
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # Compute loss
    # loss is measured from error between current and newly expected Q values
    loss = criterion(current_q_values, expected_q_values.unsqueeze(1))

    # Compute gradients (backward pass)
    loss.backward()

    # Update the parameters
    optimizer.step()

def test(episodes, clipn, save_gif=False):
    """This function will test the trained model"""
    rewards_list = []
    for i in range(episodes):
        state = env.reset()
        rewards = 0
        frames = []
        while True:
            # Save frames for gif
            if (i+1) % 100 == 0:
                env.render()
                frame = env.render(mode='rgb_array')
                frames.append(frame)

            # Select action and step to next state
            action = select_action(FloatTensor([state]), train=False)
            next_state, reward, done, _ = env.step(action[0, 0].item())

            # Update state and rewards
            state = next_state
            rewards += reward

            if done:
                break

        rewards_list.append(rewards)
    print('average reward over {} episodes: {}'.format(episodes, (sum(rewards_list)/len(rewards_list))))

    # Save gif
    if save_gif:
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_gif(clipn, fps=30)

    return rewards_list

def plot_durations(values, fign, save_fig=False):
    """This function plots the steps per episode"""
    plt.figure(figsize=(6,5))
    plt.clf()
    durations_t = torch.FloatTensor(values)
    plt.title('Testing DQN')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(durations_t.numpy())

    # Plot average line
    mean = sum(values) / len(values)
    plt.axhline(mean, c='orange', ls='--', label='average')

    if save_fig:
        plt.savefig(fign)
    else:
        plt.show()

def plot_res(values, fign, save_fig=False):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle('Training DQN')
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()

    if save_fig:
        plt.savefig(fign)
    else:
        plt.show()
# -------------------------- End Functions -------------------------- #

# ------------------------- Train & Test DQN ------------------------ #
steps_done = 0
def main():
    """Main function to execute training and testing of model"""
    save = False
    rewards_list = []
    for e in tqdm(range(EPISODES)):
        run_episode(e, env, rewards_list)

    # Plot steps per episode and histogram of results
    plot_res(rewards_list, fign='dqn_agent_train.png', save_fig=save)

    # Test model
    episodes = 100
    rewards = test(episodes, clipn='dqn_agent_test_render.gif', save_gif=save)

    # Plot rewards per episode and average
    plot_durations(rewards, fign='dqn_agent_test_avg.png', save_fig=save)

    print('Complete...')

if __name__ == "__main__":
    main()
