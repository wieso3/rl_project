import torch
from pettingzoo.magent import battle_v3
import os
import numpy as np
from DGN import DGN
from ReplayBuffer import ReplayBuffer
from utils import observation
import copy
# torch.cuda.set_device(0)

print("Starting the RL experiment!")

max_cycles = 200
map_size = 30
receptive_field = 3
n_episodes = 2000
e_before_train = 1
e_before_eps_anneal = 5
batch_size=16
feature_size = 13 * 13 * 3
GAMMA = 0.97
eps = 0.2

max_neighbors = 5

# smoothing for updating target model
tau = 0.9


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

            while len(rows) < max_neighbors and len(rows) != max_neighbors:
                rows.append(np.zeros(n_agents))
            adjacencies[name] = np.vstack(rows)
            continue

        for k, scnd_name in enumerate(red_team):

            scnd_pos = positions[scnd_name]
            if scnd_name == name or scnd_pos is None:
                continue

            if cheby_dist(cur_pos, scnd_pos) <= receptive_field:
                rows.append(eyes[k])

        if len(rows) > max_neighbors:
            rows = rows[:max_neighbors]

        while len(rows) != max_neighbors:
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

buffer = ReplayBuffer(buffer_size=5000)


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
model = DGN(n_red, feature_size, 512, 21, False)
model = model.float()
model_t = DGN(n_red, feature_size, 512, 21, False)

optimizer = torch.optim.Adam(model.parameters())

reward_to_plot = []

print(model)
blue_model = PPO.load("ppo_policy", device="cpu")
for e in range(n_episodes):
    model.train()
    obs = env.reset()

    dones = {name : False for name in original_handles}
    positions = {name : None for name in original_handles}
    adjacencies = get_adjacency(positions)
    cur_rewards = []
    cur_handles = original_handles.tolist()

    if e > e_before_eps_anneal:
        eps -= 0.0004
        if eps < 0.1:
            eps = 0.1

    for k in range(max_cycles):
        action_dict = {}

        # print(obs.keys(), len(list(obs.keys())))
        for i, name in enumerate(cur_handles):
            if name in dones and dones[name]:
                # TODO: we may need to fill in dummy values for dead agents?
                print(f"{name} is dead")
                action = None
                cur_handles.remove(name)
                positions[name] = None

            else:
                # since we use minimap mode for our adj, we only want the "normal" observations for our obs
                if name in blue_team:
                    action_dict[name] = 0 # env.action_space(name).sample() # blue_model.predict(obs[name][:,:,[0,1,2,4,5]], deterministic=True)[0]
                else:
                    if np.random.rand() < eps:
                        action_dict[name] = env.action_space(name).sample()
                    else:
                        with torch.no_grad():
                            action_dict[name] = model(observation(obs[name]), adjacencies[name], 1).argmax(1).item()

                # NOTE: the position information is always in the last two dimensions of the observation!
                positions[name] = (round(obs[name][0,0,-2] * map_size), round(obs[name][0,0,-1] * map_size))


        adjacencies = get_adjacency(positions)
        print(action_dict)
        next_obs, rewards, dones, infos = env.step(action_dict)
        print(rewards)

        if max(list(rewards.values())) > 5:
            for keyy in rewards.keys():
                if rewards[keyy] > 5:
                    print("THIS IS AN UNUSUALLY HIGH REWARD!")
                    print(keyy)
                    # print(obs[name])
                    print(rewards[keyy])
                    print(positions[keyy])
                    print(dones[keyy])

        cur_rewards.append(np.mean(list(rewards.values())))

        if k % 2 == 0:
            if len(obs) == len(next_obs):
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
            cur_r = []
            cur_a = []
            cur_d = []

            for j, name in enumerate(red_team):

                if name in s.keys():
                    cur_s.append(observation(s[name]))
                    cur_r.append(r[name])
                    cur_d.append(d[name])
                    cur_a.append(a[name] if name in a else 0)
                else:
                    cur_s.append(np.zeros(feature_size))
                    cur_r.append(0)
                    cur_d.append(True)
                    cur_a.append(0)

                if name in n_s.keys():
                    cur_ns.append(observation(s[name]))
                else:
                    cur_ns.append(np.zeros(feature_size))

                cur_adj.append(adjacencies[name])



            states.append(cur_s)
            new_states.append(cur_ns)
            adj.append(cur_adj)


            reward.append(cur_r)
            actions.append(cur_a)
            done.append(cur_d)

        optimizer.zero_grad()

        # these are of shape (num_agents * batch_size, feature_size)
        states = np.array(states).squeeze()
        new_states = np.array(new_states).squeeze()
        adj = np.array(adj).squeeze()

        reward = torch.tensor(reward).squeeze()
        actions = torch.tensor(actions).long().squeeze()
        done = torch.tensor(done).int().squeeze()


        q_values = model(states, adj)

        q_values = torch.gather(input=q_values, dim=1, index=actions.unsqueeze(1))
        t_q_values = reward + (1 - done) * GAMMA * torch.max(model_t(new_states, adj),dim=1)[0]

        loss = torch.nn.functional.mse_loss(q_values.squeeze(), t_q_values.squeeze())


        loss.backward()
        optimizer.step()
        # print(loss.item())

        # NOTE: we broadcast the feature matrix to all of our agents. the number of agents is essentially our batch size
        # we mul: [30,10,30] * [30,30,845]
        # which is [n_agents, adjacency] * [n_agents, feature_matrix]
        # this can probably be applied to our model!
        # print(torch.bmm(t_adj, torch.broadcast_to(t_s, (30, 30, 507))).shape)


        # NOTE: if we seperately execute this for each agent, it actually works!
        # print("combined", (adjacencies['red_12'] @ states[0]).shape)

        with torch.no_grad():
            for p, p_t in zip(model.parameters(), model_t.parameters()):
                p_t.data.mul_(tau)
                p_t.data.add_((1-tau) * p.data)

    mean_reward = np.mean(cur_rewards)
    reward_to_plot.append(mean_reward)
    print(f"episode {e} mean reward {mean_reward}, buf count {buffer.count()}", flush=True)

    if e % 50 == 0:
        torch.save(model.state_dict(), "model_state.zip")

import matplotlib.pyplot as plt

plt.plot(reward_to_plot)
plt.savefig("mean_reward.png")



