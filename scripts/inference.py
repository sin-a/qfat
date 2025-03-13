import logging

import hydra

from qfat.conf.configs import InferenceEntrypointCfg
from qfat.constants import CONFIG_PATH
from qfat.entrypoints.inference import InferenceEntrypoint
from qfat.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_PATH / "inference"),
    config_name="kitchen",
)
def main(cfg: InferenceEntrypointCfg) -> None:
    set_seed(cfg.seed)
    callbacks = {}
    if cfg.callbacks is not None:
        for key in cfg.callbacks.keys():
            callbacks[key] = [
                hydra.utils.instantiate(
                    clb_cfg,
                    _recursive_=False
                    if "recursive_init" not in clb_cfg
                    else clb_cfg.recursive_init,
                    _convert_="none",
                )
                for clb_cfg in cfg.callbacks[key]
            ]
    inference_ep = InferenceEntrypoint(cfg, callbacks=callbacks)
    run_id = inference_ep()
    logger.info(run_id)


if __name__ == "__main__":
    main()
