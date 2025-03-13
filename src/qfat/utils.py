import os
import random
import tempfile
from collections import Counter
from typing import Dict, Hashable, List, Tuple, Union

import hydra
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from omegaconf import OmegaConf
from torch.distributions import MixtureSameFamily

import wandb
from qfat.conf.configs import TrainingEntrypointCfg

# matplotlib.use("Agg")


def set_seed(seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_image_from_fig(fig: Figure) -> np.ndarray:
    """Turns figure into an RGB array"""
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return arr


def covariance_ellipse(mean, cov, n_std=2.0, **kwargs):
    """Returns a matplotlib Ellipse representing the covariance matrix `cov` centered at `mean`"""

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    return ellipse


def load_metric_from_wandb(
    run_id: Union[str, List[str]], metric_name: str
) -> Dict[str, List[float]]:
    """Loads a specific metric from a list of wandb runs.

    Args:
        run_id (Union[str, List[str]]): Run id(s).
        metric_name (str): The tracked quantity to load from the run.

    Returns:
        Dict[str, List[float]]: A dict mapping a run name to the time-series
            of the metric.
    """
    api = wandb.Api()
    run_metrics = {}
    if isinstance(run_id, str):
        run_id = [run_id]
    for _id in run_id:
        run = api.run(_id)
        history = run.scan_history(keys=[metric_name])
        run_metrics[_id] = [row[metric_name] for row in history]
    return run_metrics


def load_training_config(training_run_id: str) -> TrainingEntrypointCfg:
    """Loads the trainer config from a training run"""
    api = wandb.Api()
    training_run = api.run(training_run_id)
    config_file = training_run.file("hydra_config.yaml")
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file.download(
            root=tmp_dir,
            replace=True,
        )
        with open(f"{tmp_dir}/{config_file.name}", "r") as file:
            config = yaml.safe_load(file)
    return OmegaConf.create(config)


def load_model_from_wandb(training_run_id: str, model_afid: str) -> torch.nn.Module:
    """Loads a model from a training run"""
    api = wandb.Api()
    artifact = api.artifact(model_afid, type="model")
    trainer_cfg = load_training_config(training_run_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = artifact.download(root=tmp_dir)
        model = hydra.utils.instantiate(
            trainer_cfg.model_cfg, _recursive_=False, _convert_="none"
        )
        model_path = f"{artifact_dir}/model.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    return model


def get_latest_model_name(training_run_id: str, checkpoint: bool = False) -> str:
    """Retrieves the model name given the run path"""
    run_path = training_run_id.split("/")
    prefix = "checkpoint" if checkpoint else "model"
    afid = run_path[:-1] + [f"{prefix}_{run_path[-1]}:latest"]
    return "/".join(afid)


def compute_entropy(data: List[Hashable]) -> float:
    """Computes the discrete entropy of a list of hashable elements"""
    counter = Counter(data)
    total_count = len(data)
    probabilities = [counter[key] / total_count for key in counter]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy


def compute_component_variances(
    dist: MixtureSameFamily,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes how far the mixture modes are, and the total variance of individual components"""
    pi = dist.mixture_distribution.probs.unsqueeze(-1)  # shape: (B, T, K, 1)
    mu = dist.component_distribution.loc  # shape: (B, T, K, D)
    var = dist.component_distribution.variances  # shape: (B, T, K, D)
    mixture_mean = (pi * mu).sum(dim=(2)).pow(2).mean()
    weighted_mu_sq = (pi * mu.pow(2)).sum(dim=(2)).mean()
    weighted_sigma_sq = (pi * var).sum(dim=(2)).mean()
    mu_contrib = weighted_mu_sq - mixture_mean
    return mu_contrib, weighted_sigma_sq
