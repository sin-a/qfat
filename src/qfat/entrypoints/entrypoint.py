import logging
from abc import ABC, abstractmethod

from omegaconf import OmegaConf

import wandb
from qfat.conf.configs import EntrypointCfg

logger = logging.getLogger(__name__)


class Entrypoint(ABC):
    def __init__(self, cfg: EntrypointCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.run = None

    def _on_run_start(self) -> None:
        """Starts a wandb run and logs the configs to the console."""
        logger.info("Training entrypoint started.")
        logger.debug(self.cfg)
        wandb_cfg = self.cfg.wandb_cfg
        # Initialize wandb run
        self.run = wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            group=wandb_cfg.group,
            id=wandb_cfg.id,
            notes=wandb_cfg.notes,
            config=dict(self.cfg),
            resume="allow",
            mode=wandb_cfg.mode,
            settings={
                "_service_wait": 600,
                "init_timeout": 600,
                "_disable_stats": wandb_cfg.log_system_metrics,
            },
        )
        # Log config file to be able to load it for reproducibility
        path = f"{self.run.dir}/hydra_config.yaml"
        with open(path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        wandb.save(path, base_path=self.run.dir, policy="now")

    def _on_run_end(self) -> None:
        """Ends the wandb run and logs info to the console."""
        run_id = self.run.id
        logger.info("Entrypoint ended.")
        logger.info(f"Wandb run id: {run_id}")
        self.run.finish()
        return run_id

    @abstractmethod
    def _run(self) -> None:
        """Where the main code and logic should be put."""
        pass

    def __call__(self) -> str:
        self._on_run_start()
        self._run()
        return self._on_run_end()

    def __enter__(self):
        self._on_run_start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logging.error("An exception occurred: %s", exc_value, exc_info=True)
        self._on_run_end()
