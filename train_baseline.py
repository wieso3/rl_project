from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.magent import battle_v3, battlefield_v3
import supersuit as ss
import argparse


def setup_environment(env_name):
    """
    Set up an enviroment for baseline training. Some Supersuit preprocessing will be applied to them

    Args:
        env_name: name of the enviroment to train for. Choose from [battle battlefield]
    Returns:
        env: parallel MAgent env suitable for stable-baselines
    """
    if env_name == "battle":
        env = battle_v3.parallel_env(map_size=30, minimap_mode=False, step_reward=-0.005,
                                     dead_penalty=-2, attack_penalty=-0.1, attack_opponent_reward=5,
                                     max_cycles=250, extra_features=False)

    elif env_name == "battlefield":
        env = battlefield_v3.parallel_env(map_size=50, minimap_mode=False, step_reward=-0.005,
                                     dead_penalty=-2, attack_penalty=-0.1, attack_opponent_reward=5,
                                     max_cycles=450, extra_features=False)
    else:
        raise ValueError("No valid enviroment has been chosen. Please choose from [battle battlefield]!")

    env = ss.black_death_v2(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 12, 1, base_class="stable_baselines3")

    return env


def train_baseline_ppo(env, n_steps, model_name):
    """
    Train PPO baseline on a given environment and save the resulting model for later use.

    Args:
        env: MAgent parallel env suitable for training with stable-baselines
        n_steps: number of steps we want to train our model for
        model_name: the model will be saved under this name
    Returns:
        Nothing, but will save to models/model_name when training was a success.
    """
    model = PPO(MlpPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
                vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
    model.learn(total_timesteps=n_steps)
    model.save("models/" + model_name)
    print(f"Baseline model has been successfully saved to {model_name}.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO baseline on a desired enviroment with stable-baselines."+\
                                     " Note that it will require access to CUDA and will not compute on CPU!")
    parser.add_argument("-env", metavar="E", type=str, help="The MAgent environment to train for. Choose from " + \
                        "[battle battlefield]. Default: battle", default="battle")
    parser.add_argument("-steps", metavar="N", type=int, help="number of steps to train for. Default: 50000",
                        default=50000)

    args = parser.parse_args()

    env = setup_environment(args.env.lower())

    train_baseline_ppo(env, args.steps, args.env.lower() + "_ppo_policy")