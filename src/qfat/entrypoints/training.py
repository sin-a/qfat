import logging
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from qfat.callbacks.callbacks import Callback
from qfat.conf.configs import TrainingEntrypointCfg
from qfat.datasets.dataset import IMPLEMENTED_COLLATE_FNS, Batch
from qfat.entrypoints.entrypoint import Entrypoint
from qfat.models.generative_model import ModelOutput
from qfat.utils import (
    get_latest_model_name,
    load_training_config,
)

logger = logging.getLogger(__name__)


ALLOWED_CALLBACKS = [
    "epoch_start",
    "iteration_start",
    "run_start",
    "epoch_end",
    "iteration_end",
    "run_end",
]


class TrainingEntrypoint(Entrypoint):
    def __init__(
        self,
        cfg: TrainingEntrypointCfg,
        callbacks: Optional[Dict[str, List[Callback]]] = None,
    ) -> None:
        super().__init__(cfg=cfg)

        self.callbacks = {}
        if callbacks is not None:
            if not all([key in ALLOWED_CALLBACKS for key in callbacks.keys()]):
                raise ValueError(f"The allowed callback keys are {ALLOWED_CALLBACKS}")
            self.callbacks = defaultdict(list, callbacks)

    def _configure_scheduler(self) -> None:
        """Sets the scheduler with a linear warmup followed by cosine annealing with warm restarts"""

        warmup_epochs = self.model.optimizer_cfg.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.model.optimizer_cfg.n_epochs - warmup_epochs,
            eta_min=self.model.optimizer_cfg.min_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    def _get_datasets(
        self,
    ) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
        cfg: TrainingEntrypointCfg = self.cfg
        data = hydra.utils.instantiate(
            cfg.training_dataset_cfg, _recursive_=True, _convert_="none"
        )
        train_data = data
        val_data = None
        if cfg.train_ratio != 1:
            len_train = int(cfg.train_ratio * len(data))
            len_val = len(data) - len_train
            train_data, val_data = data.split(lengths=[len_train, len_val])
        if cfg.slicer_cfg is not None:
            train_data = hydra.utils.instantiate(
                cfg.slicer_cfg, dataset=train_data, _recursive_=False, _convert_="none"
            )
            if val_data is not None:
                val_data = hydra.utils.instantiate(
                    cfg.slicer_cfg,
                    dataset=val_data,
                    _recursive_=False,
                    _convert_="none",
                )
        return train_data, val_data

    def _get_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        cfg_dlt = self.cfg.train_dataloader_cfg
        train_loader = DataLoader(
            self.train_data,
            shuffle=cfg_dlt.shuffle,
            pin_memory=cfg_dlt.pin_memory,
            batch_size=cfg_dlt.batch_size,
            num_workers=cfg_dlt.num_workers,
            collate_fn=IMPLEMENTED_COLLATE_FNS[self.cfg.collate_fn_name],
            persistent_workers=cfg_dlt.persistent_workers,
        )
        val_loader = None
        if self.val_data is not None:
            cfg_dle = self.cfg.val_dataloader_cfg
            val_loader = DataLoader(
                self.train_data,
                shuffle=cfg_dle.shuffle,
                pin_memory=cfg_dle.pin_memory,
                batch_size=cfg_dle.batch_size,
                num_workers=cfg_dle.num_workers,
                collate_fn=IMPLEMENTED_COLLATE_FNS[self.cfg.collate_fn_name],
            )
        logger.info(f"Number of training batches per epoch: {len(train_loader)}")
        return train_loader, val_loader

    def _on_epoch_start(self) -> None:
        if "epoch_start" in self.callbacks.keys():
            for clb in self.callbacks["epoch_start"]:
                clb(self)

    def _on_epoch_end(self) -> None:
        cfg: TrainingEntrypointCfg = self.cfg
        if "epoch_end" in self.callbacks.keys():
            for clb in self.callbacks["epoch_end"]:
                clb(self)

        if self.val_data is not None:
            if (self.epoch % cfg.n_eval) == 0 and self.epoch != 0:
                logger.info(f"Starting model evaluation at epoch {self.epoch}")
                self.val_loss = self._compute_val_loss()
                self.run.log({"val/loss": self.val_loss})
                if self.val_loss < self.best_loss and self.cfg.log_best_model:
                    model_path = f"{self.run.dir}/model.pth"
                    torch.save(self.model.state_dict(), model_path)
                    self.run.log_model(
                        name=f"best_model_{self.run.id}",
                        path=model_path,
                        aliases=[f"epoch_{self.epoch}"],
                    )
                    self.best_loss = self.val_loss

        if (self.epoch % cfg.n_save_model) == 0 and self.epoch != 0:
            logger.info(f"Logging checkpoint to wandb at epoch {self.epoch}")
            checkpoint_path = f"{self.run.dir}/checkpoint_epoch_{self.epoch}.pth"

            checkpoint = {
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
                if self.optimizer is not None
                else None,
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
            }
            torch.save(checkpoint, checkpoint_path)
            self.run.log_artifact(
                checkpoint_path,
                name=f"checkpoint_epoch_{self.epoch}_{self.run.id}",
            )

        if self.model.optimizer_cfg.use_cosine_schedule:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr, "train/epoch": self.epoch})
        if self.cfg.device == "mps":
            torch.mps.empty_cache()
        elif self.cfg.device == "cuda":
            torch.cuda.empty_cache()

        self.epoch += 1

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        output: ModelOutput = self.model(batch)
        loss = output.loss
        loss = loss * self.loss_scaling
        return loss

    @torch.inference_mode()
    def _compute_val_loss(self) -> torch.Tensor:
        val_loss = 0
        self.model.eval()
        for batch in tqdm(self.val_loader, desc="Evaluating Model"):
            batch.to(self.cfg.device)
            val_loss += self._compute_loss(batch=batch)
        val_loss = val_loss / len(self.val_loader)
        return val_loss

    def _on_iteration_start(self) -> None:
        if "iteration_start" in self.callbacks.keys():
            for clb in self.callbacks["iteration_start"]:
                clb(self)

    def _on_iteration_end(self) -> None:
        if "iteration_end" in self.callbacks.keys():
            for clb in self.callbacks["iteration_end"]:
                clb(self)
        if self.batch_counter % self.cfg.n_train_loss == 0:
            self.run.log(
                {"train/loss": self.train_loss.detach()},
            )
        self.train_loss = None

    def load_checkpoint_from_wandb(self) -> None:
        """Loads a model, optimizer and scheduler from the given run"""
        artifact = self.run.use_artifact(
            get_latest_model_name(self.run.path, checkpoint=True)
        )
        trainer_cfg = load_training_config(self.run.path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = artifact.download(root=tmp_dir)
            self.model = hydra.utils.instantiate(
                trainer_cfg.model_cfg, _recursive_=False, _convert_="none"
            )
            checkpoint_path = f"{artifact_dir}/checkpoint.pth"
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.model.to(self.cfg.device)
            self.optimizer = self.model._configure_optimizer()
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint["scheduler"] is not None:
                self._configure_scheduler()
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if not self.cfg.reset_epoch:
                self.epoch = checkpoint["epoch"]
            if not self.cfg.reset_cfg:
                logger.info("Using the trainer config logged in the data")
                self.cfg = trainer_cfg

    def _on_run_start(self) -> None:
        super()._on_run_start()

        self.optimizer = None  # set inside _run function
        self.scheduler = None  # set inside _run function
        self.epoch = 0
        try:
            logger.info("Attempting to load the latest checkpoint from wandb...")
            self.load_checkpoint_from_wandb()
            self.model.train()
            logger.info("Checkpoint loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting from scratch.")
            self.model = hydra.utils.instantiate(
                self.cfg.model_cfg, _recursive_=False, _convert_="none"
            )

        self.runtime_transforms = []
        if self.cfg.runtime_transforms is not None:
            self.runtime_transforms = [
                hydra.utils.instantiate(transform, _recursive_=False, _convert_="none")
                for transform in self.cfg.runtime_transforms
            ]

        # if slicer_cfg is not None, then we should divide
        # the loss by the action horizon to make losses comparable
        # this will help keep the learning rate the same
        self.loss_scaling = (
            1
            if self.cfg.slicer_cfg is None
            else 1 / (1 + self.cfg.slicer_cfg.action_horizon)
        )

        self.train_loss = None
        self.val_loss = None
        self.best_loss = float("inf")
        self.batch_counter = 0
        self.global_batch_counter = 0
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
        logger.info("Logging checkpoint")
        checkpoint_path = f"{self.run.dir}/checkpoint.pth"
        checkpoint = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(checkpoint, checkpoint_path)
        self.run.log_artifact(
            checkpoint_path,
            name=f"checkpoint_{self.run.id}",
        )
        return super()._on_run_end()

    def _run(self) -> None:
        """Runs a training loop on the passed model."""
        cfg: TrainingEntrypointCfg = self.cfg
        self.train_data, self.val_data = self._get_datasets()
        self.train_loader, self.val_loader = self._get_data_loaders()
        self.model.to(cfg.device)
        if self.optimizer is None:
            self.optimizer = self.model._configure_optimizer()
        if self.scheduler is None:
            if self.model.optimizer_cfg.use_cosine_schedule:
                self._configure_scheduler()

        epochs = tqdm(
            range(self.epoch, self.model.optimizer_cfg.n_epochs), desc="Training epochs"
        )
        for _ in epochs:
            self._on_epoch_start()
            self.model.train()
            for self.batch_counter, batch in enumerate(self.train_loader):
                self._on_iteration_start()
                batch.to(self.cfg.device)
                for transform in self.runtime_transforms:
                    batch = transform(batch, self.epoch)
                train_loss = self._compute_loss(batch)
                self.model.zero_grad(set_to_none=True)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.model.optimizer_cfg.grad_norm_clip
                )
                self.optimizer.step()
                self.train_loss = train_loss.detach()
                self._on_iteration_end()
            self._on_epoch_end()
