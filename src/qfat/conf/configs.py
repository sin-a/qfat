from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

from qfat.constants import (
    ANT_DATA_PATH,
    KITCHEN_DATA_PATH,
    PUSHT_DATA_PATH,
    UR3_DATA_PATH,
)


@dataclass
class MultiheadAttentionCfg:
    embed_dim: int = MISSING  # Total dimension of the model.
    num_heads: int = MISSING  # Number of parallel attention heads.
    dropout: float = 0.0  # Dropout probability on attn_output_weights.
    bias: bool = True  # Adds bias to input/output projection layers.
    add_bias_kv: bool = False  # Adds bias to the key and value sequences at dim=0.
    add_zero_attn: bool = (
        False  # Adds a batch of zeros to the key and value sequences at dim=1.
    )
    device: Optional[str] = None  # The device to use


@dataclass
class DecoderBlockCfg:
    mlp_expansion_factor: int = 4
    residual_dropout: float = 0
    mha_cfg: MultiheadAttentionCfg = MISSING


@dataclass
class ModelCfg:
    _target_: str = MISSING


@dataclass
class EncoderCfg:
    _target_: str = MISSING


@dataclass
class IdentityEncoderCfg(EncoderCfg):
    _target_: str = "qfat.models.encoders.IdentityEncoder"


@dataclass
class HeirarchicalResNetEncoderCfg(EncoderCfg):
    _target_: str = "qfat.models.encoders.HierarchicalResNet"
    freeze_weights: bool = True


@dataclass
class OptimizerCfg:
    learning_rate: float = 3e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1  # only applied on matmul weights
    grad_norm_clip: float = 1.0
    n_epochs: int = MISSING
    use_cosine_schedule: bool = False
    warmup_epochs: int = 20
    min_lr: float = 1e-6
    encoder_lr_fraction: float = (
        0.1  # learning rate fraction for the encoder of the model
    )


@dataclass
class QFATCfg(ModelCfg):
    _target_: str = "qfat.models.qfat.QFAT"
    context_len: int = MISSING  # maximum sequence length to have at the input
    input_dim: int = MISSING  # dimensionality of the input
    n_layer: int = MISSING  # number of decoder blocks
    out_dim: int = MISSING  # output dimension
    mixture_size: int = MISSING  # number of Gaussian mixtures for each output
    embd_dropout: float = (
        MISSING  # probability of dropout after initial embedding projection
    )
    encoder_cfg: EncoderCfg = MISSING
    optimizer_cfg: OptimizerCfg = MISSING
    decoder_block_cfg: DecoderBlockCfg = MISSING
    variance_tol: float = 1e-6  # a lower bound on the variances of the mixtures
    masking_strategy: str = "causal"  # the masking to apply in the decoder blocks
    full_covariance: bool = (
        False  # whether to use a full covariance matrix or only a diagonal one.
    )
    lambda_mixtures: float = 0  # entropy regularization weight on the mixtures
    pad_sampling: bool = False  # whether to pad the sequence length during sampling to the context length
    conditional_seq_dim: Optional[int] = None
    share_state_encoder: bool = True
    share_projection_layer: bool = True
    sample_fn: str = "modes"
    goal_pos_emb: bool = False
    history_mask_prob: float = (
        0.0  # masking each element in the history of the states with this probability
    )


@dataclass
class WandbCfg:
    name: str = MISSING
    group: str = MISSING
    project: str = MISSING
    id: Optional[str] = None  # specify if you want to resume a run
    notes: Optional[str] = None
    mode: str = "online"
    log_system_metrics: bool = True


@dataclass
class TransformCfg:
    _target_: str = MISSING


class ConcatPrevActionsTransformCfg(TransformCfg):
    _target_ = "qfat.datasets.transform.ConcatPrevActionsTransform"


@dataclass
class TrajectoryDatasetCfg:
    # should be the path to torch dataset
    _target_: str = MISSING
    transforms: Optional[List] = None  # should be a list of transforms
    stats_path: Optional[str] = None
    include_goals: bool = False


@dataclass
class DummyDatasetCfg(TrajectoryDatasetCfg):
    _target_: str = "qfat.datasets.dummy.DummyDataset"
    input_dim: int = 2
    sequence_len: int = 10
    output_dim: int = 2
    num_samples: int = 1000


@dataclass
class MultiPathTrajectoryDatasetCfg(TrajectoryDatasetCfg):
    _target_: str = "qfat.datasets.multi_route.MultiPathTrajectoryDataset"
    num_samples: int = 20_000
    noise_scale: List[List[float]] = MISSING


@dataclass
class KitchenTrajectoryDatasetCfg(TrajectoryDatasetCfg):
    _target_: str = "qfat.datasets.kitchen.KitchenTrajectoryDataset"
    data_dir: str = KITCHEN_DATA_PATH
    mode: Optional[str] = None
    stats_path: str = str(KITCHEN_DATA_PATH / "data_stats.json")


@dataclass
class AntTrajectoryDatasetCfg(TrajectoryDatasetCfg):
    _target_: str = "qfat.datasets.ant.AntTrajectoryDataset"
    data_dir: str = ANT_DATA_PATH


@dataclass
class PushTTrajectoryDatasetCfg(TrajectoryDatasetCfg):
    _target_: str = "qfat.datasets.pusht.PushTTrajectoryDataset"
    data_dir: str = str(PUSHT_DATA_PATH)
    mode: str = "keypoints"  # or "embeddings" or "image"
    embedding_path: str = str(PUSHT_DATA_PATH / "resnet_embeddings")
    include_goals: bool = False
    include_prev_actions: bool = False


@dataclass
class UR3TrajectoryDatasetCfg(TrajectoryDatasetCfg):
    _target_: str = "qfat.datasets.ur3.UR3TrajectoryDataset"
    data_dir: str = UR3_DATA_PATH


@dataclass
class SlicedTrajectoryDatasetCfg:
    _target_: str = "qfat.datasets.slicer.SlicedTrajectoryDataset"
    window: int = MISSING
    action_horizon: int = 0
    min_future_sep: int = (
        0  # Minimum separation between the current window and the future goal.
    )
    future_seq_len: int = 1  # Length of the future sequence to extract as a goal.
    only_sample_tail: bool = (
        False  #  Whether to only sample the tail for the future goal sequence.
    )


@dataclass
class DataLoaderCfg:
    batch_size: int = MISSING
    num_workers: int = MISSING
    shuffle: bool = False
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class EnvCfg:
    _target_: str = MISSING


@dataclass
class MultiRouteEnvCfg(EnvCfg):
    _target_: str = "qfat.environments.multi_route.env.MultiRouteEnvV1"
    random_reset: bool = False  # whether to perform resets randomly in the environment
    track_trajectories: bool = False


@dataclass
class KitchenEnvCfg(EnvCfg):
    _target_: str = "qfat.environments.kitchen.env.KitchenEnv"
    start_from_data: bool = False
    mask_goals: bool = True  # whether to zero out the goals from the state


@dataclass
class PushTEnvCfg(EnvCfg):
    _target_: str = MISSING
    reset_to_state: Optional[List[float]] = None
    use_success_thresh: bool = False


@dataclass
class GoalConditionalEnvCfg(EnvCfg):
    _target_: str = MISSING
    env: Any = MISSING
    dataset: Any = MISSING
    future_seq_len: int = MISSING
    min_future_sep: int = MISSING
    only_sample_tail: bool = False
    random_clip_trajectory: bool = False
    use_labels_env_task: bool = False
    use_raw_goals_env_task: bool = False


@dataclass
class SamplerCfg:
    """Should contain the sampler __init__ kwargs, excluding the model"""

    _target_: str = MISSING


@dataclass
class AutoRegressiveSamplerCfg(SamplerCfg):
    _target_: str = "qfat.samplers.sampler.AutoRegressiveSampler"
    context_len: Optional[int] = None
    temperature: float = 1
    horizon: int = 0
    use_tensors: bool = False
    sample_fn: str = "modes"


@dataclass
class DualContextAutoRegressiveSamplerCfg(SamplerCfg):
    _target_: str = "qfat.samplers.sampler.DualContextAutoRegressiveSampler"
    state_context_len: Optional[int] = None
    action_context_len: Optional[int] = None
    temperature: float = 1
    use_tensors: bool = False


@dataclass
class EntrypointCfg:
    wandb_cfg: WandbCfg = MISSING
    seed: int = 0


@dataclass
class StudentForcingCfg:
    prob: float = 0.3  # the probability of using the model's prediction instead of gt


@dataclass
class TrainingEntrypointCfg(EntrypointCfg):
    device: str = "cpu"
    train_ratio: float = 0.9
    n_save_model: float = 10  # how many times per epoch, floats allowed
    model_cfg: ModelCfg = MISSING
    training_dataset_cfg: TrajectoryDatasetCfg = MISSING
    slicer_cfg: Optional[SlicedTrajectoryDatasetCfg] = None
    train_dataloader_cfg: DataLoaderCfg = MISSING
    val_dataloader_cfg: DataLoaderCfg = MISSING
    n_eval: float = 10  # how many times per epoch, floats allowed
    callbacks: Optional[Dict[str, List]] = None
    runtime_transforms: Optional[List] = None
    reset_epoch: bool = (
        True  # when reloading from a checkpoint, whether to reset epoch counter to 0
    )
    reset_cfg: bool = False  # when reloading a model from a checkpoint, whether to reset the training configs or use the logged config
    n_train_loss: int = (
        1  # every how many batches should you log the train loss to wandb.
    )
    collate_fn_name: str = "collate_policy"
    log_best_model: bool = (
        False  # wether to log the best model every n_save_model epochs
    )


@dataclass
class ConditionalEnvParams:
    labels: Optional[List[int]] = None
    n_tasks: Optional[int] = None


@dataclass
class InferenceEntrypointCfg(EntrypointCfg):
    device: str = "cpu"
    training_run_id: str = MISSING
    model_afid: str = MISSING
    sampler_cfg: SamplerCfg = MISSING
    env_cfg: EnvCfg = MISSING
    render_mode: Optional[str] = "human"  # or rgb_array or None
    num_episodes: Optional[int] = None  # if none, while loop till interrupted
    max_steps_per_episode: int = MISSING
    callbacks: Optional[Dict[str, List]] = None
    conditional_env_params: Optional[ConditionalEnvParams] = None
