import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch

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
        Nothing
    """
    rewards = []
    for name in names:
        with open("rewards/" + name, 'rb') as fp:
            reward_list = pickle.load(fp)
            rewards.append(reward_list)

    plot_rewards(rewards, plot_name, len(rewards))


def plot_rewards(rewards, name):
    """
    Plots the rewards given. If there is more than one sequence of rewards, print mean and std. of these rewards.

    Args:
        rewards: list containing equal-length lists of rewards trajectories
        name: name to use when saving the plot
    Returns:
        Nothing
    """
    n_runs = len(rewards)
    plt.clf()

    y = [i for i in range(len(rewards[0]))]

    if n_runs > 1:

        mean = np.mean(rewards, axis=0)
        std = np.std(rewards, axis=0)

        plt.plot(mean, label=r'$\mu$', color='r')
        plt.plot(mean + std, color='orange', label=r'$\mu \pm \sigma$')
        plt.plot(mean - std, color='orange')

        plt.fill_between(y, mean + std, mean - std, facecolor='red', alpha=0.1)

    else:
        plt.plot(rewards[0], label="rewards")

    plt.title(f"Mean rewards from {n_runs} runs")
    plt.ylabel("mean rewards")
    plt.xlabel("episodes")
    plt.xticks(y)
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

    plot_from_dumps(os.listdir("rewards"), "dumped_rewards")
