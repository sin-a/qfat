import logging
import re
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

import wandb
from qfat.callbacks.callbacks import Callback
from qfat.conf.configs import InferenceEntrypointCfg, TrainingEntrypointCfg
from qfat.entrypoints.entrypoint import Entrypoint
from qfat.environments.goal_conditional import GoalAppendingWrapper
from qfat.models.generative_model import ModelOutput

logger = logging.getLogger(__name__)
ALLOWED_CALLBACKS = [
    "episode_start",
    "step_start",
    "run_start",
    "episode_end",
    "step_end",
    "run_end",
]


class InferenceEntrypoint(Entrypoint):
    def __init__(
        self,
        cfg: InferenceEntrypointCfg,
        callbacks: Optional[Dict[str, List[Callback]]] = None,
    ) -> None:
        self.cfg = cfg
        self._api = wandb.Api()
        self.training_run = self._api.run(cfg.training_run_id)
        self.trainer_cfg: TrainingEntrypointCfg = self.load_training_config()
        self.model = self.load_model()
        self.env = hydra.utils.instantiate(
            cfg.env_cfg, _recursive_=True, _convert_="none"
        )
        self.sampler = hydra.utils.instantiate(
            cfg.sampler_cfg, model=self.model, _recursive_=False, _convert_="none"
        )
        self.callbacks = {}
        if callbacks is not None:
            if not all([key in ALLOWED_CALLBACKS for key in callbacks.keys()]):
                raise ValueError(f"The allowed callback keys are {ALLOWED_CALLBACKS}")
            self.callbacks = defaultdict(list, callbacks)

        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.cumulative_reward = 0
        self.current_sample_metadata = None
        self.current_model_output: ModelOutput = None
        self.episode_counter = 0
        self.episode_frames = []
        self.episode_rewards = []
        self.step = 0

    def load_training_config(self) -> TrainingEntrypointCfg:
        config_file = self.training_run.file("hydra_config.yaml")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file.download(
                root=tmp_dir,
                replace=True,
            )
            with open(f"{tmp_dir}/{config_file.name}", "r") as file:
                config = yaml.safe_load(file)
        return OmegaConf.create(config)

    def _load_model_from_checkpoint(self) -> torch.nn.Module:
        """
        Loads a *full training checkpoint* (model + optimizer + scheduler + epoch)
        but only restores the model's weights for inference.
        """
        checkpoint_artifact = self._api.artifact(
            self.cfg.model_afid,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = checkpoint_artifact.download(root=tmp_dir)
            pattern = r"checkpoint_epoch_(\d+)"
            match = re.search(pattern, self.cfg.model_afid)
            checkpoint_path = f"{artifact_dir}/{match.group(0)}.pth"

            checkpoint_data = torch.load(
                checkpoint_path, map_location=torch.device(self.cfg.device)
            )

            model_cfg = self.trainer_cfg.model_cfg
            model = hydra.utils.instantiate(
                model_cfg, _recursive_=False, _convert_="none"
            )
            model.load_state_dict(checkpoint_data["model"], strict=False)

            model.to(self.cfg.device)
            model.eval()

        return model

    def _load_model_from_state_dict(self) -> torch.nn.Module:
        artifact = self._api.artifact(self.cfg.model_afid, type="model")
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = artifact.download(root=tmp_dir)
            model = hydra.utils.instantiate(
                self.trainer_cfg.model_cfg, _recursive_=False, _convert_="none"
            )
            pattern = r"model_epoch_(\d+)"
            match = re.search(pattern, self.cfg.model_afid)
            model_path = f"{artifact_dir}/{match.group(0)}.pth"
            model.load_state_dict(
                torch.load(
                    model_path,
                    weights_only=True,
                    map_location=torch.device(self.cfg.device),
                ),
                strict=False,
            )
        model.eval()
        model.to(self.cfg.device)
        return model

    def load_model(self) -> torch.nn.Module:
        if "model" in self.cfg.model_afid:
            model = self._load_model_from_state_dict()
        elif "checkpoint" in self.cfg.model_afid:
            model = self._load_model_from_checkpoint()
        else:
            raise ValueError()
        return model

    def _on_run_start(self) -> None:
        super()._on_run_start()
        if "run_start" in self.callbacks.keys():
            for clb in self.callbacks["run_start"]:
                clb(self)

    def _on_run_end(self) -> Optional[str]:
        if "run_end" in self.callbacks.keys():
            for clb in self.callbacks["run_end"]:
                clb(self)
        for _, key_callbacks in self.callbacks.items():
            for callback in key_callbacks:
                callback.finalize()

        return super()._on_run_end()

    def _on_episode_start(self) -> None:
        if "episode_start" in self.callbacks.keys():
            for clb in self.callbacks["episode_start"]:
                clb(self)

    def _on_episode_end(self) -> None:
        if "episode_end" in self.callbacks.keys():
            for clb in self.callbacks["episode_end"]:
                clb(self)
        self.episode_counter += 1
        self.step = 0
        self.episode_frames = []
        self.episode_rewards = []
        self.cumulative_reward = 0
        self.env.close()

    def _on_step_start(self) -> None:
        if "step_start" in self.callbacks.keys():
            for clb in self.callbacks["step_start"]:
                clb(self)

    def _on_step_end(self) -> None:
        self.episode_rewards.append(self.current_reward)
        if "step_end" in self.callbacks.keys():
            for clb in self.callbacks["step_end"]:
                clb(self)
        self.step += 1

    def _run(self) -> None:
        """Rolls out a rendered episode using a trained agent with horizon handling."""
        while (
            self.cfg.num_episodes - 1 >= self.episode_counter
            if self.cfg.num_episodes is not None
            else float("inf") >= self.episode_counter
        ):
            self.current_state = self.env.reset()
            self.sampler.reset()
            done = False
            self.step = 0
            self._on_episode_start()
            state_context = [self.current_state]
            logger.info(f"Starting episode {self.episode_counter}")

            while not done and self.step < self.cfg.max_steps_per_episode:
                model_kwargs = {}
                if isinstance(self.env, GoalAppendingWrapper):
                    model_kwargs = {"conditional_seq": self.env.current_goal}
                (
                    sampled_actions,
                    self.current_model_output,
                    self.current_sample_metadata,
                ) = self.sampler.sample(x=state_context, model_kwargs=model_kwargs)

                action_chunks: List[np.ndarray] = np.split(
                    sampled_actions.squeeze(), self.sampler.horizon + 1
                )
                horizon = 0
                state_context = []
                while (
                    not done
                    and self.step < self.cfg.max_steps_per_episode
                    and horizon < self.sampler.horizon + 1
                ):
                    current_action = action_chunks[horizon]
                    self._on_step_start()
                    (
                        self.current_state,
                        self.current_reward,
                        done,
                        _,
                    ) = self.env.step(current_action)
                    state_context.append(self.current_state)
                    if self.cfg.render_mode is not None:
                        img = self.env.render(self.cfg.render_mode)
                        self.episode_frames.append(img)
                    self._on_step_end()
                    horizon += 1
                    if done and self.step >= self.cfg.max_steps_per_episode:
                        self.env.close()
            self._on_episode_end()
