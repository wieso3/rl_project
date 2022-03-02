
"""
Explanation for config values:

model options:
    use_att: if we want to use the att model or the mean model
    emb_dim: embedding dimension for observations
    receptive_field: how far our agents can "see" when generating adjacencies
    max_neighbors: max. amount of neighbors each agent can have, for more stable batching

environment options
    max_cycles: max amount of steps per episode
    map_size: the map size for the environment. Not suitable for gather
    n_actions: number of actions possible

training options:
    n_episodes: number of episodes to run
    e_before_train: number of episodes to run before starting to train
    e_before_eps_anneal: number of episodes to run before decreasing eps
    eps_anneal_factor: amount by which we multiply eps for decreasing it
    exploration_eps: starting value for eps for epsilon exploration
    batch_size: number of samples to grab for each training step
    feature_size: size of observations when flattened
    GAMMA: gamma value for Q learning
    lr: learning rate for the Adam optimizer
    tau: DEPRECATED. Percentage by which we update our target model in [0,1]

"""



battle_config = {
    "use_att" : True,
    "emb_dim" : 256,
    "receptive_field" : 6,
    "max_neighbors" : 5,

    "max_cycles" : 150,
    "map_size" : 30,
    "n_actions" : 21,

    "n_episodes" : 1200,
    "e_before_train" : 200,
    "e_before_eps_anneal" : 150,
    "eps_anneal_factor" : 0.994,
    "exploration_eps" : 0.9,

    "batch_size" : 16,
    "feature_size" : 13 * 13 * 3,
    "GAMMA" : 0.9,
    "lr" : 0.0002,
    "tau" : 0.95

}


battlefield_config = {
    "use_att" : True,
    "emb_dim" : 256,
    "receptive_field" : 9,
    "max_neighbors" : 3,

    "max_cycles" : 150,
    "map_size" : 60,
    "n_actions" : 21,

    "n_episodes" : 1201,
    "e_before_train" : 200,
    "e_before_eps_anneal" : 150,
    "eps_anneal_factor" : 0.994,
    "exploration_eps" : 0.9,

    "batch_size" : 16,
    "feature_size" : 13 * 13 * 3,
    "GAMMA" : 0.9,
    "lr" : 0.0002,
    "tau" : 0.95

}


gather_config = {
    "use_att" : True,
    "emb_dim" : 256,
    "receptive_field" : 6,
    "max_neighbors" : 5,

    "max_cycles" : 150,
    "n_actions" : 33,

    "n_episodes" : 1201,
    "e_before_train" : 200,
    "e_before_eps_anneal" : 150,
    "eps_anneal_factor" : 0.994,
    "exploration_eps" : 0.9,

    "batch_size" : 16,
    "feature_size" : 15 * 15 * 3,
    "GAMMA" : 0.9,
    "lr" : 0.0002,
    "tau" : 0.95
}
