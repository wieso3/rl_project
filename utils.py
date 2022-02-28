import numpy as np

def observation(state):
    conca = np.concatenate((state[:,:,0].flatten(), (state[:,:,1] - state[:,:,4]).flatten(), (state[:,:,2] - state[:,:,5]).flatten()))

    return conca