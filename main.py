import torch
from pettingzoo.magent import battle_v3
import os
import numpy as np
from DGN import DGN
from ReplayBuffer import ReplayBuffer
torch.cuda.set_device(0)

print("Starting the RL experiment!")

max_cycles = 200
map_size = 30
receptive_field = 3
n_episodes = 5
e_before_train = 1
batch_size=16
feature_size = 13 * 13 * 5


def get_adjacency(positions):

    # N is number of agents
    # each agent needs an adjacency matrix of size (neighbors+1 x N)
    # first row is the one hot of our current agent
    # rows j=2,..,n_neighbors are one-hot of the neighbor

    # feature matrix is of size (N x L) with L as length of feature vector


    # for each agent i: adjacency_i x feature_matrix = feature vectors in the local region


    # n_agents = len(original_handles)
    n_agents = len(red_team)

    adjacencies = {}

    def cheby_dist(pos_a, pos_b):
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))

    eyes = np.eye(N=n_agents)

    for i, name in enumerate(red_team):
        cur_pos = positions[name]

        rows = [eyes[i]]

        if cur_pos is None:
            # print("name", name, "was none in adj")

            while len(rows) < 10 and len(rows) != 10:
                rows.append(np.zeros(n_agents))
            adjacencies[name] = np.vstack(rows)
            continue

        for k, scnd_name in enumerate(red_team):

            scnd_pos = positions[scnd_name]
            if scnd_name == name or scnd_pos is None:
                continue

            if cheby_dist(cur_pos, scnd_pos) <= receptive_field:
                rows.append(eyes[k])

        if len(rows) > 10:
            rows = rows[:10]

        while len(rows) != 10:
            rows.append(np.zeros(n_agents))



        adjacencies[name] = np.vstack(rows)

    return adjacencies

# np.set_printoptions(linewidth=2000,threshold=2000)



from stable_baselines3 import PPO, DQN
import supersuit as ss

# env = battle_v3.parallel_env(map_size=map_size, minimap_mode=False, step_reward=-0.005,
#     dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
#     max_cycles=max_cycles, extra_features=False)
# env = ss.black_death_v2(env)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 12, 1, base_class='stable_baselines3')
#
#
# from stable_baselines3.dqn import MlpPolicy
# model = DQN(MlpPolicy, env, learning_starts=1000, verbose=2)
# model.learn(total_timesteps=max_cycles)
# model.save("dqn_policy")

# from stable_baselines3.ppo import MlpPolicy
# model = PPO(MlpPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
# model.learn(total_timesteps=max_cycles)
# model.save("ppo_policy")


env = battle_v3.parallel_env(map_size=map_size, minimap_mode=True, max_cycles=max_cycles, extra_features=False)
n_red, n_blue = env.team_sizes
handles = env.agents
agents = np.array(env.agents)
original_handles = np.copy(handles)


# env = ss.black_death_v2(env)






# TODO work out the following function: this works out some feature vector for a given agent
#   flatten may definitely be of use since it mashes all features into a 1d array
#   then we can get started on incorporating the model!
def observation(state1,state2):
    state = []
    for j in range(20):
        state.append(np.hstack(((state1[j][0:11,0:11,1]-state1[j][0:11,0:11,5]).flatten(),state2[j][-1:-3:-1])))
    return state

buffer = ReplayBuffer(buffer_size=200000)


red_index = np.arange(0,n_red)
blue_index = np.arange(n_red, n_red + n_blue)


red_team = agents[red_index]
blue_team = agents[blue_index]

# obs channels (battle):
# index | obs channel content
# ---------------------------
#   0   | obstacle/off the map
#   1   | my_team_presence
#   2   | my_team_hp
#   3   | other_team_presence
#   4   | other_team_hp

# model = DQN.load("dqn_policy", device="cpu")
model = PPO.load("ppo_policy", device="cpu")
for e in range(n_episodes):

    obs = env.reset()

    dones = {name : False for name in original_handles}
    positions = {name : None for name in original_handles}
    cur_rewards = []

    for k in range(max_cycles):
        action_dict = {}
        for i, name in enumerate(handles):
            if dones[name]:
                # TODO: we may need to fill in dummy values for dead agents?
                print(f"{name} is dead")
                action = None
                handles.remove(name)
                positions[name] = None

            else:
                # since we use minimap mode for our adj, we only want the "normal" observations for our obs
                cur_obs = obs[name][:,:,[0,1,2,4,5]]

                if name in red_team:
                    action = model.predict(cur_obs, deterministic=True)[0]
                else:
                    action = env.action_space(name).sample()

                # NOTE: the position information is always in the last two dimensions of the observation!
                positions[name] = (round(obs[name][0,0,-2] * map_size), round(obs[name][0,0,-1] * map_size))

            if action is not None:
                action_dict[name] = action

        adjacencies = get_adjacency(positions)
        next_obs, rewards, dones, infos = env.step(action_dict)
        if max(list(rewards.values())) > 5:
            for keyy in rewards.keys():
                if rewards[keyy] > 5:
                    print("THIS IS AN UNUSUALLY HIGH REWARD!")
                    print(keyy)
                    # print(obs[name])
                    print(rewards[keyy])
                    print(positions[keyy])
                    print(dones[keyy])

        # print(adjacencies['red_26'].shape)
        cur_rewards.append(np.mean(list(rewards.values())))

        if k % 3 == 0:
            buffer.add(obs, action_dict, rewards, next_obs, dones, adjacencies)

        obs = next_obs


        env.render()

        if e < e_before_train:
            continue
        if k % 3 != 0:
            continue

        # TRAINING #
        # print(f"Train in e {e} step {k}")

        batch = buffer.get_batch(batch_size=1)
        states, actions, reward, new_states, done, adj = [], [], [], [], [], []

        for i, b_item in enumerate(batch):

            s, a, r, n_s, d, ad = b_item[0], b_item[1], b_item[2], b_item[3], b_item[4], b_item[5]
            cur_s = []
            cur_ns = []
            cur_adj = []
            for j, name in enumerate(red_team):
                if name in s.keys():
                    cur_s.append(s[name][:,:,[0,1,2,4,5]].flatten())
                else:
                    cur_s.append(np.zeros(feature_size))

                if name in n_s.keys():
                    cur_ns.append(n_s[name][:,:,[0,1,2,4,5]].flatten())
                else:
                    cur_ns.append(np.zeros(feature_size))

                cur_adj.append(adjacencies[name])
                # print(adjacencies[name].shape)

            states.append(cur_s)
            new_states.append(cur_ns)
            adj.append(cur_adj)


        # these are of shape (num_agents * batch_size, feature_size)
        states = np.array(states).squeeze()

        new_states = np.array(new_states).squeeze()
        adj = np.array(adj).squeeze()

        print("-----\n")
        print("adj", adj.shape)
        print("states", states.shape)


        t_adj = torch.tensor(adj)
        t_s = torch.tensor(states).double()

        print(t_adj.shape)
        print(t_s.shape)

        # NOTE: we broadcast the feature matrix to all of our agents. the number of agents is essentially our batch size
        # we mul: [30,10,30] * [30,30,845]
        # which is [n_agents, adjacency] * [n_agents, feature_matrix]
        # this can probably be applied to our model!
        print(torch.bmm(t_adj, torch.broadcast_to(t_s, (30, 30, 845))).shape)

        # NOTE: if we seperately execute this for each agent, it actually works!
        # print("combined", (adjacencies['red_12'] @ states[0]).shape)


        # TODO: input state and adjacency into the model and look whats working from there
        #       maybe start with an averaging kernel and get attention working after that

        # TODO: grab Q values from model
        #       and target Q values from target model
        #       (depending on done we either only have reward, or also have future states
        #       loss - optim - step

        # TODO: to update the target model
        #       NOTE: the target model is of same shape as the normal one
        #       Update target by grabbing the weights from both: w_t = beta * w_n + (1 - beta) * w_t  (3.1 in paper)

    print(f"episode {e} mean reward {np.mean(cur_rewards)}")


    # TODO: we may want to save our MODEL!


