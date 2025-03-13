import logging

import hydra

from qfat.conf.configs import TrainingEntrypointCfg
from qfat.constants import CONFIG_PATH
from qfat.entrypoints.training import TrainingEntrypoint
from qfat.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH / "training"), config_name="kitchen"
)
def main(cfg: TrainingEntrypointCfg) -> None:
    set_seed(cfg.seed)
    callbacks = {}
    if cfg.callbacks is not None:
        for key in cfg.callbacks.keys():
            callbacks[key] = [
                hydra.utils.instantiate(clb_cfg, _recursive_=False, _convert_="none")
                for clb_cfg in cfg.callbacks[key]
            ]
    trainer_ep = TrainingEntrypoint(cfg, callbacks=callbacks)
    with trainer_ep as ep:
        ep._run()


if __name__ == "__main__":
    main()
