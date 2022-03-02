import torch
from pettingzoo.magent import gather_v4
import argparse
import numpy as np
from models.DGN import DGN
from utils import ReplayBuffer, observation, load_torch_model
import matplotlib.pyplot as plt
import pickle
# torch.cuda.set_device(0)


def get_adjacency(positions):
    """
    Construct the adjacency matrices for a set of given positions. Note that all agents will have a position. If their
    position is None, then their adjacency will reflect them only having access to their OWN feature values.

    For a given agent, their adjacency is of shape (max_neighbors + 1, n_agents). The first row contains the
    index of the original agent. The following (#max_neighbors) rows contain indices of agents that this agent is
    connected to. If there aren't enough neighbors, we pad with zeros.

    Two agents are considered connected, if the Chebyshev-Distance of their positions is <= receptive_field.

    We can use these adjacencies to multiply with the feature matrix of shape (n_agents, feature_size) to get all
    features of neighbors of shape (max_neighbors + 1, feature_size).

    Args:
        positions: dictionary containing positions for every agent. These are None for dead agents and a tuple (int,int)
                   otherwise.
    Returns:
        adjacencies: dictionary containing adjacencies of shape (max_neighbors + 1, n_agents) for each agent
    """

    team = original_handles
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


def show_match(model):
    """
    Execute a show match for one episode and rendering it.

    Args:
        model: the model to use for the omnivores
    Returns:
        Nothing, but renders the episode being played
    """
    model.eval()
    dones = {name: False for name in original_handles}
    positions = {name: None for name in original_handles}
    adjacencies = get_adjacency(positions)
    cur_handles = original_handles.tolist()
    cur_rewards = []

    for k in range(max_cycles):
        action_dict = {}
        if len(env.agents) <= 1:
            # this is an emergency break, because segmentation faults occur when all agents are done
            break

        for i, name in enumerate(cur_handles):
            # handling dead agents
            if (name in dones and dones[name]) or name not in obs:
                cur_handles.remove(name)
                positions[name] = None

            else:
                # since we use minimap mode for our adj, we only want the "normal" observations for our obs

                if np.random.rand() < 0.1:
                    action_dict[name] = env.action_space(name).sample()
                else:
                    with torch.no_grad():
                        action_dict[name] = model(observation(obs[name]), adjacencies[name], 1).argmax(1).item()

                if name in obs:
                    positions[name] = (round(obs[name][0, 0, -2] * map_size), round(obs[name][0, 0, -1] * map_size))
                else:
                    positions[name] = None


        adjacencies = get_adjacency(positions)
        next_obs, rewards, dones, infos = env.step(action_dict)
        env.render()

        cur_rewards.append(np.mean(list(rewards.values())))

    print(np.mean(cur_rewards))



def train_model(model, model_t):
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
                # handling dead agents
                if (name in dones and dones[name]) or name not in obs:
                    cur_handles.remove(name)
                    positions[name] = None

                else:
                    # since we use minimap mode for our adj, we only want the "normal" observations for our obs
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
            next_obs, rewards, dones, infos = env.step(action_dict)

            cur_rewards.append(np.mean(list(rewards.values())))

            buf_dict = {}
            for name in original_handles:
                to_append = []
                if name in obs:
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

                for j, name in enumerate(original_handles):
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

        # after 50 episodes, update the target model
        if e % 50 == 0:
            with torch.no_grad():
                for p, p_t in zip(model.parameters(), model_t.parameters()):
                    p_t.data.mul_(0)
                    p_t.data.add_(1 * p.data)


    print("Saving the model to model/" + env_name + "model_state.zip.")
    torch.save(model.state_dict(), "models/" + env_name + "model_state.zip")
    print("Plotting rewards and losses!")
    plt.clf()
    plt.plot(reward_to_plot)
    plt.savefig("mean_rewards_" + run_name + ".png")

    plt.clf()
    plt.plot(losses)
    plt.savefig("loss/loss_" + run_name + ".png")

    buffer.erase()

    return reward_to_plot


# obs channels (gather):
# index | obs channel content
# ---------------------------
#   0   | obstacle/off the map
#   1   | omnivore_presence
#   2   | omnivore_hp
#   3   | food_presence
#   4   | food_hp


if __name__ == '__main__':

    from config import gather_config

    parser = argparse.ArgumentParser(description="Execute or train a model on one of the battle MAgent environments" + \
                                     " Hyperparameters are provided via the config.py file!.")
    parser.add_argument('-show', dest='show', action='store_const',
                    const=True, default=False,help='If True, execute a show match (with rendering activated).')
    parser.add_argument("-seed", metavar="S", type=int, help="Seed for the current experiment. Default: 1",
                        default=1)

    args = parser.parse_args()

    print("Starting the RL experiment!")

    env_name = "gather"

    config = gather_config
    max_cycles = config['max_cycles']
    env = gather_v4.parallel_env(minimap_mode=True, max_cycles=max_cycles, extra_features=False,
                                 dead_penalty=-2, attack_penalty=-0.01)

    all_rewards = []

    # seeds = [1,2,3,4,5] # 1


    print(f"Running the enviroment {env_name}.")
    print("Hyperparams follow:")
    print(config)

    seed = args.seed

    map_size = 200
    use_att = config["use_att"]
    emb_dim = config["emb_dim"]
    receptive_field = config["receptive_field"]
    max_neighbors = config["max_neighbors"]
    n_actions = config["n_actions"]

    n_episodes = config["n_episodes"]
    e_before_train = config["e_before_train"]
    e_before_eps_anneal = config["e_before_eps_anneal"]
    exploration_eps = config["exploration_eps"]
    eps_anneal_factor = config["eps_anneal_factor"]
    batch_size = config["batch_size"]
    feature_size = config["feature_size"]
    GAMMA = config["GAMMA"]
    lr = config["lr"]

    run_name = env_name + "_" + str(seed)

    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    handles = env.agents
    agents = np.array(env.agents)
    original_handles = np.copy(handles)
    n_agents = len(agents)

    if args.show:
        print("With show being True, we will execute one show match!")
        model = DGN(n_agents, feature_size, emb_dim, n_actions, use_att=use_att)
        model = load_torch_model(model, "models/" + env_name + "_model_state.zip")
        show_match(model)
        exit()


    model = DGN(n_agents, feature_size, emb_dim, n_actions, use_att=use_att)
    model = model.float()
    model_t = DGN(n_agents, feature_size, emb_dim, n_actions, use_att=use_att)

    rewards = train_model(model, model_t)


    with open("rewards/" + run_name, 'wb') as fp:
        pickle.dump(rewards, fp)