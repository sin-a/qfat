import logging
from typing import List, Literal, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import wandb
from qfat.callbacks.callbacks import Callback
from qfat.conf.configs import EnvCfg, SamplerCfg
from qfat.entrypoints.training import TrainingEntrypoint
from qfat.environments.goal_conditional import GoalAppendingWrapper
from qfat.models.generative_model import ModelOutput
from qfat.utils import compute_component_variances

logger = logging.getLogger(__name__)


class ValidationVarianceCallback(Callback):
    def __init__(self, frequency: int):
        """Computes and logs variance metrics on the validation dataset.

        Args:
            frequency (int): How often (in epochs) to run this validation metric computation.
        """
        self.frequency = frequency

    def __call__(self, ep: "TrainingEntrypoint"):
        if ep.epoch % self.frequency == 0 and ep.epoch != 0:
            if ep.val_data is None or ep.val_loader is None:
                logger.warning(
                    "No validation data provided, cannot compute validation variance."
                )
                return

            logger.info(
                f"Computing variance metrics on validation set at epoch {ep.epoch}"
            )

            ep.model.eval()

            total_mu_contrib = 0.0
            total_weighted_sigma_sq = 0.0
            count = 0

            with torch.inference_mode():
                for batch in ep.val_loader:
                    batch.to(ep.cfg.device)
                    output: ModelOutput = ep.model(batch)
                    dist = ep.model.get_distribution(output.output)
                    mu_contrib, weighted_sigma_sq = compute_component_variances(dist)
                    total_mu_contrib += mu_contrib.item()
                    total_weighted_sigma_sq += weighted_sigma_sq.item()
                    count += 1
                avg_mu_contrib = total_mu_contrib / count
                avg_weighted_sigma_sq = total_weighted_sigma_sq / count
                wandb.log(
                    {"val/mixture_distances": avg_mu_contrib},
                )
                wandb.log(
                    {"val/weighted_mixture_variance": avg_weighted_sigma_sq},
                )


class RolloutCallback(Callback):
    def __init__(
        self,
        n_rollouts: int,
        max_steps: int,
        env_cfg: EnvCfg,
        sampler_cfg: SamplerCfg,
        frequency: int,
        reward_reduction: Literal["mean", "max", "last", "sum"] = "sum",
        log_entropy: bool = False,
        filter_tasks_entropy: Optional[List[str]] = None,
        is_goal_conditional: bool = False,
        min_sequence_length: int = 1,
        max_sequence_length: int = 1,
        skip: int = 0,
        **kwargs,
    ):
        """Rolls out the model in an environment and computes the average collected rewards per episode.

        Args:
            n_rollouts (int): The number of rollouts to execute.
            max_steps (int): The maximum number of steps in the environment.
            env_cfg (EnvCfg): The environment configuration.
            sampler_cfg (SamplerCfg): The sampler configuration.
            frequency (int): Epoch frequency at which to evaluate the average rewards.
            reward_reduction (Literal["mean", "max", "last"]): How to aggregate the episode
                rewards.
            filter_tasks_entropy (Optional[List[str]]): Task strings to filter out from the entropy
                computations.
            is_goal_conditional (bool): Whether this is a goal conditional environment or not.
            min_sequence_length (int): The minimum number of task-completion sequence length to be considered
                in the entropy computations. Defaults to 1.
            max_sequence_length (int, optional): The maximum task-completion sequence length to be considered in the
                entropy computations. Defaults to 1.
            skip (int, optional): How many epochs to skip before invoking the callback during training. Defaults to 0.
        """
        self.env_cfg = env_cfg
        self.sampler_cfg = sampler_cfg
        self.n_rollouts = n_rollouts
        self.max_steps = max_steps
        self.frequency = frequency
        self.completed_tasks = []
        self.reward_reduction = reward_reduction
        self.log_entropy = log_entropy
        self.filter_tasks_entropy = filter_tasks_entropy or []
        self.is_goal_conditional = is_goal_conditional
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.skip = skip
        super().__init__()

    def reduce_episode_rewards(self, episode_rewards: List[float]) -> float:
        if self.reward_reduction == "mean":
            val = sum(episode_rewards) / len(episode_rewards)
        elif self.reward_reduction == "max":
            val = max(episode_rewards)
        elif self.reward_reduction == "last":
            val = episode_rewards[-1]
        elif self.reward_reduction == "sum":
            val = sum(episode_rewards)
        return val

    def __call__(self, ep: TrainingEntrypoint):
        if ep.epoch >= self.skip:
            if ep.epoch % self.frequency == 0 and ep.epoch != 0:
                logger.info("Rolling out model on the environment")
                self.env = hydra.utils.instantiate(
                    self.env_cfg, _recursive_=True, _convert_="none"
                )
                if self.log_entropy:
                    if not hasattr(self.env, "completed_tasks"):
                        raise Exception(
                            "Environment must have a 'completed_tasks' attribute when log_entropy is True."
                        )
                if self.is_goal_conditional:
                    if not isinstance(self.env, GoalAppendingWrapper):
                        raise Exception(
                            "For goal conditional tasks, the environment must be an instance of the class GoalAppendingWrapper"
                        )
                self.sampler = hydra.utils.instantiate(
                    self.sampler_cfg,
                    model=ep.model,
                    _recursive_=False,
                    _convert_="none",
                )
                episodes_reward = 0
                for _ in tqdm(range(self.n_rollouts), desc="Model Rollouts"):
                    s = self.env.reset()
                    self.sampler.reset()
                    done = False
                    step = 0
                    episode_reward = []
                    s_context = [s]
                    while not done and step < self.max_steps:
                        model_kwargs = {}
                        if self.is_goal_conditional:
                            model_kwargs = {"conditional_seq": self.env.current_goal}
                        a, *_ = self.sampler.sample(
                            x=s_context, model_kwargs=model_kwargs
                        )
                        a: List[np.ndarray] = np.split(
                            a.squeeze(), self.sampler.horizon + 1
                        )
                        horizon = 0
                        s_context = []
                        while (
                            not done
                            and step < self.max_steps
                            and horizon < self.sampler.horizon + 1
                        ):
                            a_h = a[horizon]
                            s, r, *_ = self.env.step(a_h)
                            episode_reward.append(r)
                            s_context.append(s)
                            step += 1
                            horizon += 1

                    if self.log_entropy:
                        task = self.env.completed_tasks
                        if len(task) >= self.min_sequence_length:
                            if len(task) > self.max_sequence_length:
                                task = task[: self.max_sequence_length + 1]
                            task_str = "->".join(task)
                            self.completed_tasks.append(task_str)

                    episodes_reward += self.reduce_episode_rewards(
                        episode_rewards=episode_reward
                    )
                average_reward = episodes_reward / self.n_rollouts
                wandb.log(
                    {f"val/average_{self.reward_reduction}_reward": average_reward}
                )

                if self.log_entropy:
                    df = pd.DataFrame({"completed_task": self.completed_tasks})
                    df = df.value_counts().reset_index()
                    df["probability"] = df["count"] / df["count"].sum()
                    entropy = (
                        -(df["probability"].values * np.log2(df["probability"].values))
                        .sum()
                        .item()
                    )
                    wandb.log(
                        {
                            "val/task_entropy": entropy,
                        }
                    )
                self.completed_tasks = []
                self.sampler = None
                self.env.close()
                self.env = None
