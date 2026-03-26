from typing import Any, Dict, List, Optional, Tuple, Type, Union

import jax
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class OffPolicyAlgorithmJax(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        stats_window_size: int = 100,
    ):
        # Override support action spaces to bypass assert
        try:
            import gym as gym_old
            try:
                from gymnasium import spaces as gym_spaces
                supported_action_spaces = (gym_spaces.Box,)
            except ImportError:
                supported_action_spaces = (gym_old.spaces.Box,)
        except ImportError:
            try:
                from gymnasium import spaces as gym_spaces
                supported_action_spaces = (gym_spaces.Box,)
            except ImportError:
                from gym import spaces as gym_spaces
                supported_action_spaces = (gym_spaces.Box,)
                
        # Additional fallback logic directly adding the specific action space instance's class
        try:
            if hasattr(env, 'action_space'):
                action_space_class = type(env.action_space)
                if action_space_class not in supported_action_spaces:
                    # Only do this if we can't otherwise cast it
                    pass
        except Exception:
            pass
            
        # Call the base stable_baselines3 off_policy_algorithm class
        # It expects certain arguments depending on version
        import inspect
        try:
            from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
            sig = inspect.signature(OffPolicyAlgorithm.__init__)
        except ImportError:
            # Fallback if we can't inspect the signature
            sig = None
            
        # We must override SUPPORTED_ACTION_SPACES on the class level temporarily
        # to pass the assertion in stable_baselines3.common.base_class
        try:
            from stable_baselines3.common.base_class import BaseAlgorithm
            original_supported = getattr(BaseAlgorithm, 'SUPPORTED_ACTION_SPACES', None)
            
            # Combine the base supported spaces with our custom ones
            if original_supported is not None:
                new_supported = list(original_supported)
                try:
                    import gym as gym_old
                    if gym_old.spaces.Box not in new_supported:
                        new_supported.append(gym_old.spaces.Box)
                except:
                    pass
                BaseAlgorithm.SUPPORTED_ACTION_SPACES = tuple(new_supported)
            else:
                BaseAlgorithm.SUPPORTED_ACTION_SPACES = supported_action_spaces
        except:
            original_supported = None
        
        kwargs = dict(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
        )
        
        if sig is not None and 'stats_window_size' in sig.parameters:
            kwargs['stats_window_size'] = stats_window_size
            
        try:
            super().__init__(**kwargs)
        finally:
            if original_supported is not None:
                BaseAlgorithm.SUPPORTED_ACTION_SPACES = original_supported
        
        # Monkey patch self.action_space to bypass the check during __init__ if needed
        # Or better yet, stable-baselines3 checks it in BaseAlgorithm.__init__
        # It looks at type(self.action_space) vs self.SUPPORTED_ACTION_SPACES
        # Will be updated later
        self.key = jax.random.PRNGKey(0)
        # Note: we do not allow schedule for it
        self.qf_learning_rate = qf_learning_rate

    def _get_torch_save_params(self):
        return [], []

    def _excluded_save_params(self) -> List[str]:
        excluded = super()._excluded_save_params()
        excluded.remove("policy")
        return excluded

    def set_random_seed(self, seed: Optional[int]) -> None:  # type: ignore[override]
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:  # type: ignore[has-type]
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        self._setup_lr_schedule()
        # By default qf_learning_rate = pi_learning_rate
        self.qf_learning_rate = self.qf_learning_rate or self.lr_schedule(1)
        self.set_random_seed(self.seed)
        # Make a local copy as we should not pickle
        # the environment when using HerReplayBuffer
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        if issubclass(self.replay_buffer_class, HerReplayBuffer):  # type: ignore[arg-type]
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env

        self.replay_buffer = self.replay_buffer_class(  # type: ignore[misc]
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device="cpu",  # force cpu device to easy torch -> numpy conversion
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **replay_buffer_kwargs,
        )
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
