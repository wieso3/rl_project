import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
from collections import deque
import random
import argparse

def observation(state):
    """
    Returns a flattened representation of a given state.

    Args:
        state: observation as it was returned by the environment
    Returns:
        conca: 1d representation of the state
    """
    conca = np.concatenate((state[:,:,0].flatten(), (state[:,:,1] - state[:,:,4]).flatten(), (state[:,:,2] - state[:,:,5]).flatten()))

    return conca



def plot_from_dumps(names, plot_name):
    """
    Grab rewards that have been dumped in rewards/ and plot them.

    Args:
        names: names of the dumped rewards to plot
        plot_name: name to use when saving the plot
    Returns:
        Nothing, but will try to save a plot at "plot_name".
    """
    rewards = []
    for name in names:
        with open("rewards/" + name, 'rb') as fp:
            reward_list = pickle.load(fp)
            rewards.append(reward_list)
    plot_rewards(rewards, plot_name)


def plot_rewards(rewards, name, running_average_smoothing=True):
    """
    Plots the rewards given. If there is more than one sequence of rewards, print mean and std. of these rewards.

    Args:
        rewards: list containing equal-length lists of rewards trajectories
        name: name to use when saving the plot
        running_average_smoothing: whether we want to smooth the curves using running averages. Only used and
                                   useful for more than one run. Default: True
    Returns:
        Nothing, but saves a plot at "name".
    """
    n_runs = len(rewards)
    plt.clf()


    min_len = min([len(a) for a in rewards])
    x = [i for i in range(min_len)]

    for i, r in enumerate(rewards):
        rewards[i] = r[:min_len]

    if n_runs > 1:

        mean = np.mean(rewards, axis=0)
        std = np.std(rewards, axis=0)


        # running average smoothing for less noisy graphs
        if running_average_smoothing:
            kernel = np.ones(n_runs) / n_runs
            mean = np.convolve(mean, kernel, mode='same') # [:-1]
            std = np.convolve(std, kernel, mode='same')

        plt.plot(mean, label=r'$\mu$', color='r')
        plt.plot(mean + std, color='orange', label=r'$\mu \pm \sigma$')
        plt.plot(mean - std, color='orange')

        plt.fill_between(x, mean + std, mean - std, facecolor='red', alpha=0.1)

    else:
        plt.plot(rewards[0], label="rewards")

    plt.title(f"Mean rewards from {n_runs} runs")
    plt.ylabel("mean rewards")
    plt.xlabel("episodes")
    plt.legend()
    plt.savefig(name + ".png")


def load_torch_model(model, name):
    """
    Load a model from a given state_dict and return it.

    Args:
         model: the pytorch model to save the state_dict in. Needs to be of the same size
         name: the path/file name under which the state_dict is saved
    Returns:
         model: the model with a loaded state_dict and in evaluation mode
    """
    model.load_state_dict(torch.load(name))
    model.eval()

    return model


if __name__ == '__main__':

    current_dumps = os.listdir("rewards")
    parser = argparse.ArgumentParser(description="All rewards dumped in 'rewards/' will be plotted in a graph. " + \
                                     f"Currently there are {len(current_dumps)} dumps namely: {current_dumps}.")
    parser.parse_args()
    plot_from_dumps(current_dumps, "dumped_rewards")


class ReplayBuffer(object):

    """
    Standard Replay Buffer Implementation
    """

    def __init__(self, buffer_size):
        """
        Initialized the Replay Buffer

        Args:
            buffer_size: max amount of samples to store
        """
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        """
        Grabs a batch of a given size. If not that many samples are available, return all available samples.

        Args:
            batch_size: number of SARS samples
        Returns:
            batch: a batch of the desired size
        """
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        """
        Returns the max size of the replay buffer.

        Returns:
            size: number of samples that can be stored within the buffer
        """
        return self.buffer_size

    def add(self, buf_dict):
        """
        Adds a dictionary filled with values to the replay buffer. If the amount of samples will surpass the max
        amount of possible samples, the oldest sample will be discarded.

        Args:
            buf_dict: dictionary containing e.g. (S,A,R,S) tuples, adjacencies and more for each agent
        """
        experience = buf_dict
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        """
        Returns the number of samples currently in the buffer.

        Returns:
            num_experiences: number of samples currently stored
        """
        return self.num_experiences

    def erase(self):
        """
        Deletes all the current samples (via Pythons garbage collection). Sets the number of experiences back to zero
        """
        self.buffer = deque()
        self.num_experiences = 0