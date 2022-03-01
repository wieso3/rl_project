import torch
from pettingzoo.magent import battle_v3, battlefield_v3
import os
import numpy as np
from DGN import DGN
from ReplayBuffer import ReplayBuffer
from utils import observation, plot_rewards, load_torch_model
import copy
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import PPO

# torch.cuda.set_device(0)



print("Starting the RL experiment!")

def get_adjacency(positions, only_red=True):
    """
    Construct the adjacency matrices for a set of given positions. Note that all agents will have a position. If their
    position is None, then their adjacency will reflect them only having access to their OWN feature values.

    For a given agent, their adjacency is of shape (max_neighbors + 1, n_agents). The first row contains the
    index of the original agent. The following (#max_neighbors) rows contain indices of agents that this agent is
    connected to. Only agents within the same team will be connected with each other! If there aren't enough neighbors,
    we pad with zeros.

    Two agents are considered connected, if the Chebyshev-Distance of their positions is <= receptive_field.

    We can use these adjacencies to multiply with the feature matrix of shape (n_agents, feature_size) to get all
    features of neighbors of shape (max_neighbors + 1, feature_size).

    Args:
        positions: dictionary containing positions for every agent. These are None for dead agents and a tuple (int,int)
                   otherwise.
        only_red: whether we currently construct adjacencies for the red team (i.e. the one we train with).
                  Default: True
    Returns:
        adjacencies: dictionary containing adjacencies of shape (max_neighbors + 1, n_agents) for each agent in the
                     desired team
    """

    team = red_team if only_red else blue_team

    n_agents = len(team)

    adjacencies = {}

    def cheby_dist(pos_a, pos_b):
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))

    eyes = np.eye(N=n_agents)

    for i, name in enumerate(team):
        cur_pos = positions[name]

        rows = [eyes[i]]

        if cur_pos is None:

            while len(rows) < max_neighbors and len(rows) != max_neighbors:
                rows.append(np.zeros(n_agents))
            adjacencies[name] = np.vstack(rows)
            continue

        for k, scnd_name in enumerate(team):

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




# obs channels (battle):
# index | obs channel content
# ---------------------------
#   0   | obstacle/off the map
#   1   | my_team_presence
#   2   | my_team_hp
#   3   | other_team_presence
#   4   | other_team_hp

# model = DQN.load("dqn_policy", device="cpu")


def train_model(model, model_t, enemy_model):
    """
    Train a model using a target model. The enemies get simulated by a stable-baselines enemy_model.

    Args:
        model: the model to use for predictions and training
        model_t: the target model to use for smoother training purposes
        enemy_model: stable-baselines model that simulates the agents that we do not train
    Returns:
        rewards: list of mean rewards per episode
    """
    buffer = ReplayBuffer(buffer_size=7500)
    global exploration_eps
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    reward_to_plot = []
    losses = []


    for e in range(n_episodes):
        model.train()

        _ = env.reset()
        obs = env.reset()

        dones = {name : False for name in original_handles}
        positions = {name : None for name in original_handles}
        adjacencies = get_adjacency(positions)
        # blue_adjacencies = get_adjacency(positions, only_red=False)


        cur_rewards = []
        cur_handles = original_handles.tolist()

        if e > e_before_eps_anneal:
            exploration_eps *= eps_anneal_factor
            exploration_eps = max(0.1, exploration_eps)


        for k in range(max_cycles):
            action_dict = {}
            if len(env.agents) <= 1:
                # this is an emergency break, because segmentation faults occur when all agents are done
                break

            for i, name in enumerate(cur_handles):
                if (name in dones and dones[name]) or name not in obs:
                    # TODO: we may need to fill in dummy values for dead agents?
                    # print(f"{name} is dead")
                    # action_dict[name] = None
                    cur_handles.remove(name)
                    positions[name] = None

                else:
                    # since we use minimap mode for our adj, we only want the "normal" observations for our obs
                    if name in blue_team:
                        action_dict[name] = enemy_model.predict(obs[name][:,:,[0,1,2,4,5]], deterministic=True)[0]
                        # # blue_model(observation(obs[name]), blue_adjacencies[name], 1).argmax(1).item()  # 0 #env.action_space(name).sample() #
                    else:
                        if np.random.rand() < exploration_eps:
                            action_dict[name] = env.action_space(name).sample()
                        else:
                            with torch.no_grad():
                                action_dict[name] = model(observation(obs[name]), adjacencies[name], 1).argmax(1).item()

                    # NOTE: the position information is always in the last two dimensions of the observation!

                    if name in obs:
                        positions[name] = (round(obs[name][0,0,-2] * map_size), round(obs[name][0,0,-1] * map_size))
                    else:
                        positions[name] = None

            adjacencies = get_adjacency(positions)
            # blue_adjacencies = get_adjacency(positions, only_red=False)
            next_obs, rewards, dones, infos = env.step(action_dict)

            n_red_rewards = 0
            red_rewards = 0
            for name in red_team:
                if name in rewards:
                    n_red_rewards += 1
                    red_rewards += rewards[name]

            cur_rewards.append(red_rewards / n_red_rewards)


            buf_dict = {}
            for name in red_team:
                to_append = []
                if name in obs:
                    # TODO: check if dead reds need to be checked for any further!
                    to_append.append(observation(obs[name]))
                    to_append.append(action_dict[name] if name in action_dict else 0)
                    to_append.append(rewards[name] if name in rewards else 0)
                    to_append.append(observation(next_obs[name]) if name in next_obs else np.zeros(feature_size))
                    to_append.append(dones[name] if name in dones else True)
                    to_append.append(adjacencies[name])
                else:
                    to_append.append(np.zeros(feature_size))
                    to_append.append(0)
                    to_append.append(0)
                    to_append.append(np.zeros(feature_size))
                    to_append.append(True)
                    to_append.append(adjacencies[name])
                buf_dict[name] = to_append
            buffer.add(buf_dict)

            obs = next_obs


            # env.render()

            if e < e_before_train:
                continue
            if k % 8 != 0:
                continue

            # TRAINING #

            batch = buffer.get_batch(batch_size=batch_size)
            states, actions, reward, new_states, done, adj = [], [], [], [], [], []

            for i, b_item in enumerate(batch):

                buf_dict = b_item
                cur_s = []
                cur_ns = []
                cur_adj = []
                cur_r = []
                cur_a = []
                cur_d = []

                for j, name in enumerate(red_team):
                    o, a, r, n_o, d, adj_m = buf_dict[name]
                    cur_s.append(o)
                    cur_ns.append(n_o)
                    cur_adj.append(adj_m)
                    cur_r.append(r)
                    cur_a.append(a)
                    cur_d.append(d)

                states.append(cur_s)
                new_states.append(cur_ns)
                adj.append(cur_adj)


                reward.append(cur_r)
                actions.append(cur_a)
                done.append(cur_d)

            optimizer.zero_grad()

            states = np.concatenate(states, axis=0)
            new_states = np.concatenate(new_states, axis=0)
            adj = np.concatenate(adj, axis=0)

            reward = torch.tensor(np.concatenate(reward,axis=0)).float()
            actions = torch.tensor(np.concatenate(actions, axis=0)).long()
            done = torch.tensor(np.concatenate(done, axis=0)).long()

            # print("states", states.shape, "new_states", new_states.shape)
            # print("adj", adj.shape)
            # print("reward", reward.shape, "actions", actions.shape, "dones", done.shape)

            q_values = model(states, adj, n_agents=states.shape[0])

            q_values = torch.gather(input=q_values, dim=1, index=actions.unsqueeze(1))
            t_q_values = reward + (1 - done) * GAMMA * torch.max(model_t(new_states, adj, n_agents=states.shape[0]),dim=1)[0]

            # loss = torch.nn.functional.mse_loss(q_values.squeeze(), t_q_values.squeeze())
            loss = torch.nn.functional.huber_loss(q_values.squeeze(), t_q_values.squeeze())

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print("loss", loss.item())


        mean_reward = np.mean(cur_rewards)
        reward_to_plot.append(mean_reward)
        print(f"episode {e} mean reward {mean_reward}, buf count {buffer.count()}, eps {exploration_eps}", flush=True)

        if e % 50 == 0:
            with torch.no_grad():
                for p, p_t in zip(model.parameters(), model_t.parameters()):
                    p_t.data.mul_(0)
                    p_t.data.add_(1 * p.data)


        if e % 100 == 0:
            torch.save(model.state_dict(), "model_state.zip")
            plt.clf()
            plt.plot(reward_to_plot)
            plt.savefig("mean_rewards_" + run_name + ".png")

            plt.clf()
            plt.plot(losses)
            plt.savefig("loss/loss_" + run_name + ".png")

    buffer.erase()

    return reward_to_plot


all_rewards = []

seeds = [1] # [1,2,3,4,5] # 1

for seed in seeds:
    use_att = True
    emb_dim = 256
    max_cycles = 150
    map_size = 30
    receptive_field = 6
    n_episodes = 1201
    e_before_train = 200
    e_before_eps_anneal = 150
    batch_size = 16
    feature_size = 13 * 13 * 3
    GAMMA = 0.9
    lr = 0.0002
    exploration_eps = 0.9
    eps_anneal_factor = 0.994
    max_neighbors = 5
    # smoothing for updating target model
    tau = 0.95
    n_actions = 21
    run_name = str(seed)

    env = battle_v3.parallel_env(map_size=map_size, minimap_mode=True, max_cycles=max_cycles, extra_features=False,
                                 attack_opponent_reward=5, dead_penalty=-2, attack_penalty=-0.01)


    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    n_red, n_blue = env.team_sizes
    handles = env.agents
    agents = np.array(env.agents)
    original_handles = np.copy(handles)
    red_index = np.arange(0, n_red)
    blue_index = np.arange(n_red, n_red + n_blue)

    red_team = agents[red_index]
    blue_team = agents[blue_index]

    model = DGN(n_red, feature_size, emb_dim, n_actions, use_att=use_att)
    model = model.float()
    model_t = DGN(n_red, feature_size, emb_dim, n_actions, use_att=use_att)

    blue_model = PPO.load("ppo_policy", device="cpu")
    # blue_model = load_torch_model(DGN(n_red, feature_size, emb_dim, 21, use_att=use_att), "model_state_1.zip")
    rewards = train_model(model, model_t, blue_model)


    with open("rewards/" + run_name, 'wb') as fp:
        pickle.dump(rewards, fp)


# all_rewards.append(rewards)


# plot_rewards(all_rewards, "tmp")
