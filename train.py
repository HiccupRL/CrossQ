import argparse
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import jax
import jax.numpy as jnp
import rlax
import flax.linen as nn

from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from sbx import SAC
from sbx.sac.actor_critic_evaluation_callback import CriticBiasCallback, EvalCallback
from sbx.sac.utils import *

import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS

try:
    import d4rl
except ImportError:
    pass

os.environ.setdefault('PYOPENGL_PLATFORM', 'glfw')
# Configure MuJoCo to use EGL renderer
os.environ.setdefault('MUJOCO_GL', 'glfw')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['WANDB_DIR'] = '/tmp'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'

parser = argparse.ArgumentParser()
parser.add_argument("-env",         type=str, required=False, default="HumanoidStandup-v4", help="Set Environment.")
parser.add_argument("-algo",        type=str, required=True, default='sac', choices=['crossq', 'sac', 'redq', 'droq', 'td3'], help="algorithm to use (essentially a named hyperparameter set for the base SAC algorithm)")
parser.add_argument("-seed",        type=int, required=False, default=1, help="Set Seed.")
parser.add_argument("-log_freq",    type=int, required=False, default=300, help="how many times to log during training")

parser.add_argument('-wandb_entity', type=str, required=False, default=None, help='your wandb entity name')
parser.add_argument('-wandb_project', type=str, required=False, default='crossQ', help='wandb project name')
parser.add_argument("-wandb_mode",    type=str, required=False, default='disabled', choices=['disabled', 'online'], help="enable/disable wandb logging")
parser.add_argument("-eval_qbias",    type=int, required=False, default=0, choices=[0,1], help="enable/diasble q bias evaluation (expensive; experiments will run much slower)")

parser.add_argument("-adam_b1",           type=float, required=False, default=0.5, help="adam b1 hyperparameter")
parser.add_argument("-bn",                type=float, required=False, default=False,  choices=[0,1], help="Use batch norm layers in the actor and critic networks")
parser.add_argument("-bn_momentum",       type=float, required=False, default=0.99, help="batch norm momentum parameter")
parser.add_argument("-bn_mode",           type=str,   required=False, default='brn_actor', help="batch norm mode (bn / brn / brn_actor). brn_actor also uses batch renorm in the actor network")
parser.add_argument("-critic_activation", type=str,   required=False, default='relu', help="critic activation function")
parser.add_argument("-crossq_style",      type=float, required=False, default=1,choices=[0,1], help="crossq style joint forward pass through critic network")
parser.add_argument("-dropout",           type=int,   required=False, default=0, choices=[0,1], help="whether to use dropout for SAC")
parser.add_argument("-ln",                type=float, required=False, default=False, choices=[0,1], help="layernorm in critic network")
parser.add_argument("-lr",                type=float, required=False, default=1e-3, help="actor and critic learning rate")
parser.add_argument("-n_critics",         type=int,   required=False, default=2, help="number of critics to use")
parser.add_argument("-n_neurons",         type=int,   required=False, default=256, help="number of neurons for each critic layer")
parser.add_argument("-policy_delay",      type=int,   required=False, default=1, help="policy is updated after this many critic updates")
parser.add_argument("-tau",               type=float, required=False, default=0.005, help="target network averaging")
parser.add_argument("-utd",               type=int,   required=False, default=1, help="number of critic updates per env step")
parser.add_argument("-total_timesteps",   type=int,   required=False, default=5e6, help="total number of training steps (online)")
parser.add_argument("-offline_timesteps", type=int,   required=False, default=0, help="total number of offline pre-training steps")
parser.add_argument("-eval_freq",         type=int, required=False, default=1000, help="how many offline steps between evaluations")
parser.add_argument("-num_eval_episodes", type=int, required=False, default=50, help="number of episodes to evaluate")

parser.add_argument("-bnstats_live_net",  type=int,   required=False, default=0,choices=[0,1], help="use bn running statistics from live network within the target network")

experiment_time = time.time()
args = parser.parse_args()

seed = args.seed
args.algo = str.lower(args.algo)
args.bn = bool(args.bn)
args.crossq_style = bool(args.crossq_style)
args.tau = float(args.tau) if not args.crossq_style else 1.0
args.bn_momentum = float(args.bn_momentum) if args.bn else 0.0
dropout_rate, layer_norm = None, False
policy_q_reduce_fn = jax.numpy.min
net_arch = {'pi': [256, 256], 'qf': [args.n_neurons, args.n_neurons]}

total_timesteps = int(args.total_timesteps)
offline_timesteps = int(args.offline_timesteps)
eval_freq = int(args.eval_freq)
num_eval_episodes = int(args.num_eval_episodes)
log_freq = int(args.log_freq)

if 'dm_control' in args.env:
    total_timesteps = {
        'dm_control/reacher-easy'     : 100_000,
        'dm_control/reacher-hard'     : 100_000,
        'dm_control/ball_in_cup-catch': 200_000,
        'dm_control/finger-spin'      : 500_000,
        'dm_control/fish-swim'        : 5_000_000,
        'dm_control/humanoid-stand'   : 5_000_000,
    }.get(args.env, total_timesteps)
    eval_freq = max(total_timesteps // args.log_freq, 1)

elif 'antmaze' in args.env:
    total_timesteps = 1_000_000
    offline_timesteps = 1_000_000
    eval_freq = max(total_timesteps // args.log_freq, 1)

td3_mode = False

if args.algo == 'droq':
    dropout_rate = 0.01
    layer_norm = True
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 2
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'DroQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'redq':
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 10
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'REDQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'td3':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    td3_mode = True
    layer_norm = args.ln
    if args.dropout: 
        dropout_rate = 0.01
    group = f'TD3_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'sac':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    layer_norm = args.ln
    if args.dropout: 
        dropout_rate = 0.01
    group = f'SAC_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'crossq':
    args.adam_b1 = 0.5
    args.policy_delay = 3
    args.n_critics = 2
    args.utd = 1                    # nice
    net_arch["qf"] = [2048, 2048]   # wider critics
    args.bn = True                  # use batch norm
    args.bn_momentum = 0.99
    args.crossq_style = True        # with a joint forward pass
    args.tau = 1.0                  # without target networks
    group = f'CrossQ_{args.env}'

else:
    raise NotImplemented

args_dict = vars(args)
args_dict.update({
    "dropout_rate": dropout_rate,
    "layer_norm": layer_norm
})

with wandb.init(
    entity=args.wandb_entity,
    project=args.wandb_project,
    name=f"{args.env}_seed={seed}",
    group=group,
    tags=[],
    sync_tensorboard=True,
    config=args_dict,
    settings=wandb.Settings(start_method="fork") if is_slurm_job() else None,
    mode=args.wandb_mode
) as wandb_run:
    
    # SLURM maintainance
    if is_slurm_job():
        print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
        wandb_run.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')

    class AntMazeSuccessWrapper(gym.Wrapper):
        def reset(self, **kwargs):
            # Some old gym/D4RL environments don't support seed/options in reset
            try:
                return self.env.reset(**kwargs)
            except TypeError as e:
                if 'seed' in str(e) or 'options' in str(e):
                    # Strip kwargs and try again
                    return self.env.reset()
                raise e
                
        def step(self, action):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False

            if reward > 0.0:
                info['is_success'] = 1.0
            elif 'is_success' not in info:
                info['is_success'] = 0.0
                
            # If we need to adjust reward for online training like FQL
            if 'antmaze' in env_id and ('diverse' in env_id or 'play' in env_id or 'umaze' in env_id):
                reward = reward - 1.0

            if len(result) == 5:
                return obs, reward, terminated, truncated, info
            else:
                return obs, reward, terminated, info

    try:
        import d4rl
    except ImportError:
        pass
        
    try:
        import minari
    except ImportError:
        pass

    try:
        import d4rl
    except ImportError:
        pass

    # Handle standard Gym and D4RL formats
    env_id = args.env
    if 'antmaze' in env_id and '-v' not in env_id:
        env_id = env_id + '-v2' # default to v2 for D4RL

    import gymnasium as gym
    def make_env_fallback(env_id, **kwargs):
        import gymnasium as gym
        try:
            return gym.make(env_id, **kwargs)
        except Exception as gym_err:
            try:
                import gym as gym_old
                try:
                    import d4rl
                except Exception as d4rl_err:
                    print(f"[Warning] Failed to import d4rl: {d4rl_err}")
                return gym_old.make(env_id, **kwargs)
            except Exception as e:
                print(f"Error creating environment {env_id}: {e}")
                raise

    training_env = make_env_fallback(env_id)

    if 'antmaze' in env_id:
        # Standardize to gymnasium spaces if using old gym to bypass SB3 check
        try:
            from gymnasium import spaces as gym_spaces
            import gym as gym_old
            if isinstance(training_env.action_space, gym_old.spaces.Box):
                old_as = training_env.action_space
                training_env.action_space = gym_spaces.Box(low=old_as.low, high=old_as.high, shape=old_as.shape, dtype=old_as.dtype)
            if isinstance(training_env.observation_space, gym_old.spaces.Box):
                old_os = training_env.observation_space
                training_env.observation_space = gym_spaces.Box(low=old_os.low, high=old_os.high, shape=old_os.shape, dtype=old_os.dtype)
        except:
            pass
        training_env = AntMazeSuccessWrapper(training_env)
        
    # Re-extract obs_space after potential wrapper
    if hasattr(training_env, 'observation_space'):
        obs_space = training_env.observation_space
    else:
        obs_space = training_env.env.observation_space

    # Allow custom observation space adjustments
    if hasattr(training_env, 'observation_space'):
        obs_space = training_env.observation_space
    else:
        obs_space = training_env.env.observation_space

    if args.env == 'dm_control/humanoid-stand':
        try:
            obs_space['head_height'] = gym.spaces.Box(-np.inf, np.inf, (1,))
        except:
            import gym as gym_old
            obs_space['head_height'] = gym_old.spaces.Box(-np.inf, np.inf, (1,))
    if args.env == 'dm_control/fish-swim':
        try:
            obs_space['upright'] = gym.spaces.Box(-np.inf, np.inf, (1,))
        except:
            import gym as gym_old
            obs_space['upright'] = gym_old.spaces.Box(-np.inf, np.inf, (1,))

    import optax
    try:
        from gymnasium import spaces as gym_spaces
        is_dict = isinstance(obs_space, gym_spaces.Dict) or isinstance(obs_space, dict)
    except:
        try:
            import gym as gym_old
            is_dict = isinstance(obs_space, gym_old.spaces.Dict) or isinstance(obs_space, dict)
        except:
            is_dict = isinstance(obs_space, dict)

    model = SAC(
        "MultiInputPolicy" if is_dict else "MlpPolicy",
        training_env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[args.critic_activation],
            'layer_norm': layer_norm,
            'batch_norm': bool(args.bn),
            'batch_norm_momentum': float(args.bn_momentum),
            'batch_norm_mode': args.bn_mode,
            'dropout_rate': dropout_rate,
            'n_critics': args.n_critics,
            'net_arch': net_arch,
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': args.adam_b1,
                'b2': 0.999 # default
            })
        }),
        gradient_steps=args.utd,
        policy_delay=args.policy_delay,
        crossq_style=bool(args.crossq_style),
        td3_mode=td3_mode,
        use_bnstats_from_live_net=bool(args.bnstats_live_net),
        policy_q_reduce_fn=policy_q_reduce_fn,
        learning_starts=5000,
        learning_rate=args.lr,
        qf_learning_rate=args.lr,
        tau=args.tau,
        gamma=0.99 if not args.env == 'Swimmer-v4' else 0.9999,
        verbose=0,
        buffer_size=max(1_000_000, offline_timesteps * 2),
        seed=seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
    )

    # Create log dir where evaluation results will be saved
    eval_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/eval/"
    qbias_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/qbias/"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(qbias_log_dir, exist_ok=True)

    # Create callback that evaluates agent
    wrapper_class = AntMazeSuccessWrapper if 'antmaze' in args.env else None
    
    # Define a helper to wrap our env creator
    def _make_eval_env():
        env = make_env_fallback(env_id)
        if wrapper_class is not None:
            env = wrapper_class(env)
        return env
        
    try:
        gym.spec(env_id)
        # Use string ID if registered properly in gymnasium
        eval_env = make_vec_env(env_id, n_envs=1, seed=seed, wrapper_class=wrapper_class)
        qbias_eval_env = make_vec_env(env_id, n_envs=1, seed=seed, wrapper_class=wrapper_class)
    except Exception:
        # Use DummyVecEnv with the creator function directly
        def _standardize_eval_env():
            env = _make_eval_env()
            try:
                from gymnasium import spaces as gym_spaces
                import gym as gym_old
                if isinstance(env.action_space, gym_old.spaces.Box):
                    old_as = env.action_space
                    env.action_space = gym_spaces.Box(low=old_as.low, high=old_as.high, shape=old_as.shape, dtype=old_as.dtype)
                if isinstance(env.observation_space, gym_old.spaces.Box):
                    old_os = env.observation_space
                    env.observation_space = gym_spaces.Box(low=old_os.low, high=old_os.high, shape=old_os.shape, dtype=old_os.dtype)
            except:
                pass
            return env
            
        eval_env = DummyVecEnv([_standardize_eval_env])
        qbias_eval_env = DummyVecEnv([_standardize_eval_env])

    # Ensure environments are wrapped with VecMonitor for evaluation
    if not is_vecenv_wrapped(eval_env, VecMonitor):
        eval_env = VecMonitor(eval_env)
    if not is_vecenv_wrapped(qbias_eval_env, VecMonitor):
        qbias_eval_env = VecMonitor(qbias_eval_env)

    eval_callback = EvalCallback(
        eval_env,
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=eval_log_dir, eval_freq=eval_freq,
        n_eval_episodes=1, deterministic=True, render=False
    )

    # Callback that evaluates q bias according to the REDQ paper.
    q_bias_callback = CriticBiasCallback(
        qbias_eval_env, 
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=qbias_log_dir, eval_freq=eval_freq,
        n_eval_episodes=1, render=False
    )

    callback_list = CallbackList(
        [eval_callback, q_bias_callback, WandbCallback(verbose=0,)] if args.eval_qbias else 
        [eval_callback, WandbCallback(verbose=0,)]
    )

    if offline_timesteps > 0:
        print(f"Starting offline pre-training for {offline_timesteps} steps...")
        
        # Modify D4RL reward behavior if needed, to match FQL logic
        if 'antmaze' in args.env and ('diverse' in args.env or 'play' in args.env or 'umaze' in args.env):
            # The reference FQL code adjusts the reward: reward = reward - 1.0 for antmaze tasks
            adjust_reward = True
        else:
            adjust_reward = False
            
        # Populate replay buffer with offline dataset
        try:
            import os
            # Provide an option to clear broken datasets
            dataset_path = training_env.unwrapped.dataset_filepath if hasattr(training_env.unwrapped, 'dataset_filepath') else None
            try:
                # We should use qlearning_dataset to get next_observations
                import d4rl
                dataset = d4rl.qlearning_dataset(training_env)
            except OSError as e:
                if 'truncated file' in str(e) or 'Unable to synchronously open file' in str(e):
                    print(f"Warning: Dataset file corrupted. Attempting to redownload.")
                    if dataset_path and os.path.exists(dataset_path):
                        os.remove(dataset_path)
                    import d4rl
                    dataset = d4rl.qlearning_dataset(training_env)
                else:
                    raise
            except Exception as e:
                # Fallback to get_dataset and manually construct next_observations
                try:
                    raw_dataset = training_env.unwrapped.get_dataset()
                    dataset = {
                        'observations': raw_dataset['observations'][:-1],
                        'next_observations': raw_dataset['observations'][1:],
                        'actions': raw_dataset['actions'][:-1],
                        'rewards': raw_dataset['rewards'][:-1],
                        'terminals': raw_dataset['terminals'][:-1],
                        'timeouts': raw_dataset['timeouts'][:-1] if 'timeouts' in raw_dataset else np.zeros_like(raw_dataset['terminals'][:-1])
                    }
                except:
                    raise e
        except (AttributeError, ImportError, NameError):
            try:
                import d4rl
                dataset = d4rl.qlearning_dataset(training_env)
            except (NameError, ImportError, AttributeError):
                # D4RL is not available, try using minari
                try:
                    import minari
                    # Convert standard D4RL env name to Minari dataset name
                    # e.g., antmaze-umaze-v2 -> antmaze-umaze-v2 (or similar)
                    print(f"[Info] Attempting to load Minari dataset for {env_id}")
                    minari_dataset = minari.load_dataset(env_id)
                    # Minari dataset structure is different, we need to extract transitions
                    dataset = {
                        'observations': [],
                        'actions': [],
                        'next_observations': [],
                        'rewards': [],
                        'terminals': [],
                        'timeouts': []
                    }
                    for episode in minari_dataset.episodes:
                        dataset['observations'].extend(episode.observations[:-1])
                        dataset['next_observations'].extend(episode.observations[1:])
                        dataset['actions'].extend(episode.actions)
                        dataset['rewards'].extend(episode.rewards)
                        dataset['terminals'].extend(episode.terminations)
                        dataset['timeouts'].extend(episode.truncations)
                    
                    for k in dataset:
                        dataset[k] = np.array(dataset[k])
                except Exception as e:
                    print(f"Failed to load dataset: {e}")
                    raise

        # Replay buffer format:
        # observations, actions, next_observations, rewards, terminals
        N = dataset['rewards'].shape[0]
        if model.replay_buffer.buffer_size < N:
            print(f"[Warning] Replay buffer size ({model.replay_buffer.buffer_size}) is smaller than dataset size ({N}).")

        import tqdm
        batch_size = model.batch_size
        gradient_steps = model.gradient_steps
        
        # Populate buffer is slow, let's add a progress bar
        print(f"Populating replay buffer with {N} transitions...")
        # Optimize buffer population by avoiding the slow `.add` loop and directly copying arrays
        
        # Calculate how many transitions to copy
        n_transitions = min(N, model.replay_buffer.buffer_size)
        
        # Set pos and full flag
        model.replay_buffer.pos = n_transitions % model.replay_buffer.buffer_size
        model.replay_buffer.full = n_transitions >= model.replay_buffer.buffer_size
        
        # Adjust rewards if needed
        rewards = dataset['rewards'][:n_transitions]
        if adjust_reward:
            rewards = rewards - 1.0
        
        # Directly assign arrays for maximum speed
        # Stable-Baselines3 buffer expects shape (buffer_size, n_envs, dim)
        model.replay_buffer.observations[:n_transitions, 0, :] = dataset['observations'][:n_transitions]
        model.replay_buffer.next_observations[:n_transitions, 0, :] = dataset['next_observations'][:n_transitions]
        model.replay_buffer.actions[:n_transitions, 0, :] = dataset['actions'][:n_transitions]
        model.replay_buffer.rewards[:n_transitions, 0] = rewards
        model.replay_buffer.dones[:n_transitions, 0] = dataset['terminals'][:n_transitions]
        
        if 'timeouts' in dataset:
            model.replay_buffer.timeouts[:n_transitions, 0] = dataset['timeouts'][:n_transitions]
            
        print("Buffer populated. Starting offline gradient steps...")
        # We also need to set up logger
        from stable_baselines3.common.utils import configure_logger
        from stable_baselines3.common.logger import Logger, make_output_format

        try:
            # Add wandb to the logger formats
            loggers = [
                make_output_format("stdout", "offline"),
            ]
            if args.wandb_mode != 'disabled':
                # Custom wandb format for SB3
                from wandb.integration.sb3 import WandbCallback
                from stable_baselines3.common.logger import KVWriter
                
                class WandbOutputFormat(KVWriter):
                    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
                        wandb.log(key_values, step=step)
                    def close(self) -> None:
                        pass
                loggers.append(WandbOutputFormat())
                
            try:
                if model.tensorboard_log:
                    loggers.append(make_output_format("tensorboard", model.tensorboard_log))
            except Exception:
                pass
                
            model.set_logger(Logger("offline", loggers))
        except Exception as e:
            print(f"Custom logger setup failed: {e}. Falling back to default.")
            try:
                model.set_logger(configure_logger(model.verbose, model.tensorboard_log, "offline"))
            except ImportError:
                print("Tensorboard not installed, logging to stdout instead")
                model.set_logger(configure_logger(model.verbose, None, "offline"))

        import time
        from collections import defaultdict
        
        # We define a simple evaluation function similar to FQL
        def evaluate(env, model, num_eval_episodes=10):
            stats = defaultdict(list)
            for _ in range(num_eval_episodes):
                try:
                    obs = env.reset()
                except TypeError:
                    # Fallback for some wrapper structures that still pass kwargs
                    try:
                        obs = env.unwrapped.reset()
                    except:
                        raise
                # Handle tuple returned by gymnasium reset
                if isinstance(obs, tuple):
                    obs = obs[0]
                done = False
                episode_reward = 0.0
                episode_length = 0
                is_success = 0.0
                while not done:
                    # Deterministic action selection for evaluation
                    action, _ = model.predict(obs, deterministic=True)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                        
                    episode_reward += reward
                    episode_length += 1
                    if info.get('is_success', 0.0) > 0:
                        is_success = 1.0
                        
                stats['return'].append(episode_reward)
                stats['length'].append(episode_length)
                stats['success'].append(is_success)
                
            # Normalize score for D4RL
            if hasattr(env.unwrapped, 'get_normalized_score'):
                normalized_scores = [env.unwrapped.get_normalized_score(r) * 100.0 for r in stats['return']]
                stats['normalized_score'] = normalized_scores
                
            return {k: np.mean(v) for k, v in stats.items()}

        eval_env = make_env_fallback(env_id)
        if 'antmaze' in env_id:
            eval_env = AntMazeSuccessWrapper(eval_env)

        for step in tqdm.tqdm(range(1, offline_timesteps + 1)):
            model.train(batch_size=batch_size, gradient_steps=gradient_steps)
            
            # Logging training metrics
            if step % log_freq == 0:
                model.logger.record("time/iterations", step)
                model.logger.dump(step=step)
                
            # Evaluation similar to FQL
            if step % eval_freq == 0:
                eval_stats = evaluate(eval_env, model, num_eval_episodes=num_eval_episodes)
                
                # Log to SB3 logger
                for k, v in eval_stats.items():
                    model.logger.record(f"eval/{k}", v)
                model.logger.record("time/iterations", step)
                model.logger.dump(step=step)
                
                print(f"--- Offline Step {step} Eval ---")
                for k, v in eval_stats.items():
                    print(f"  {k}: {v:.3f}")
                print("--------------------------------")

    print(f"Starting online training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_list)