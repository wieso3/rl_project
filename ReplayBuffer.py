from collections import deque
import random

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
        # Randomly sample batch_size examples
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