from hydra.core.config_store import ConfigStore

from qfat.conf.configs import (
    AntTrajectoryDatasetCfg,
    AutoRegressiveSamplerCfg,
    ConcatPrevActionsTransformCfg,
    DataLoaderCfg,
    DecoderBlockCfg,
    DualContextAutoRegressiveSamplerCfg,
    DummyDatasetCfg,
    GoalConditionalEnvCfg,
    HeirarchicalResNetEncoderCfg,
    IdentityEncoderCfg,
    InferenceEntrypointCfg,
    KitchenEnvCfg,
    KitchenTrajectoryDatasetCfg,
    MultiheadAttentionCfg,
    MultiPathTrajectoryDatasetCfg,
    MultiRouteEnvCfg,
    OptimizerCfg,
    PushTEnvCfg,
    PushTTrajectoryDatasetCfg,
    QFATCfg,
    SamplerCfg,
    SlicedTrajectoryDatasetCfg,
    TrainingEntrypointCfg,
    TransformCfg,
    UR3TrajectoryDatasetCfg,
    WandbCfg,
)

CONFIG_STORE = ConfigStore.instance()

CONF_PACKAGE = "qfat.conf"
PROVIDER = "qfat"

MODEL_GROUP = "models"
OPTIMIZER_GROUP = "optimizers"
DATASET_GROUP = "datasets"
DATALOADER_GROUP = "dataloaders"
WANDB_GROUP = "wandb"
ENTRYPOINT_GROUP = "entrypoints"
SAMPLER_GROUP = "samplers"
ENVIRONMENT_GROUP = "envs"
TRANSFORMS_GROUP = "transforms"


def register_all():
    # models
    CONFIG_STORE.store(
        name=IdentityEncoderCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=MODEL_GROUP,
        node=IdentityEncoderCfg,
    )
    CONFIG_STORE.store(
        name=HeirarchicalResNetEncoderCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=MODEL_GROUP,
        node=HeirarchicalResNetEncoderCfg,
    )

    CONFIG_STORE.store(
        name=MultiheadAttentionCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=MODEL_GROUP,
        node=MultiheadAttentionCfg,
    )
    CONFIG_STORE.store(
        name=DecoderBlockCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=MODEL_GROUP,
        node=DecoderBlockCfg,
    )

    CONFIG_STORE.store(
        name=QFATCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=MODEL_GROUP,
        node=QFATCfg,
    )

    # optimizers
    CONFIG_STORE.store(
        name=OptimizerCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=OPTIMIZER_GROUP,
        node=OptimizerCfg,
    )
    # transforms
    CONFIG_STORE.store(
        name=TransformCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=TRANSFORMS_GROUP,
        node=TransformCfg,
    )
    CONFIG_STORE.store(
        name=ConcatPrevActionsTransformCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=TRANSFORMS_GROUP,
        node=ConcatPrevActionsTransformCfg,
    )
    # datasets
    CONFIG_STORE.store(
        name=DummyDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=DummyDatasetCfg,
    )
    CONFIG_STORE.store(
        name=SlicedTrajectoryDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=SlicedTrajectoryDatasetCfg,
    )
    CONFIG_STORE.store(
        name=MultiPathTrajectoryDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=MultiPathTrajectoryDatasetCfg,
    )
    CONFIG_STORE.store(
        name=KitchenTrajectoryDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=KitchenTrajectoryDatasetCfg,
    )
    CONFIG_STORE.store(
        name=PushTTrajectoryDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=PushTTrajectoryDatasetCfg,
    )
    CONFIG_STORE.store(
        name=AntTrajectoryDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=AntTrajectoryDatasetCfg,
    )
    CONFIG_STORE.store(
        name=UR3TrajectoryDatasetCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATASET_GROUP,
        node=UR3TrajectoryDatasetCfg,
    )
    # dataloaders
    CONFIG_STORE.store(
        name=DataLoaderCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=DATALOADER_GROUP,
        node=DataLoaderCfg,
    )
    # wandb
    CONFIG_STORE.store(
        name=WandbCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=WANDB_GROUP,
        node=WandbCfg,
    )
    # samplers
    CONFIG_STORE.store(
        name=SamplerCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=SAMPLER_GROUP,
        node=SamplerCfg,
    )
    CONFIG_STORE.store(
        name=AutoRegressiveSamplerCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=SAMPLER_GROUP,
        node=AutoRegressiveSamplerCfg,
    )
    CONFIG_STORE.store(
        name=DualContextAutoRegressiveSamplerCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=SAMPLER_GROUP,
        node=DualContextAutoRegressiveSamplerCfg,
    )

    # environments
    CONFIG_STORE.store(
        name=MultiRouteEnvCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=ENVIRONMENT_GROUP,
        node=MultiRouteEnvCfg,
    )

    CONFIG_STORE.store(
        name=KitchenEnvCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=ENVIRONMENT_GROUP,
        node=KitchenEnvCfg,
    )
    CONFIG_STORE.store(
        name=PushTEnvCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=ENVIRONMENT_GROUP,
        node=PushTEnvCfg,
    )
    CONFIG_STORE.store(
        name=GoalConditionalEnvCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=ENVIRONMENT_GROUP,
        node=GoalConditionalEnvCfg,
    )

    # entrypoints
    CONFIG_STORE.store(
        name=TrainingEntrypointCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=ENTRYPOINT_GROUP,
        node=TrainingEntrypointCfg,
    )
    CONFIG_STORE.store(
        name=InferenceEntrypointCfg.__name__,
        package=CONF_PACKAGE,
        provider=PROVIDER,
        group=ENTRYPOINT_GROUP,
        node=InferenceEntrypointCfg,
    )
