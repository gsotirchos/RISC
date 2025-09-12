from hive.utils.loggers import WandbLogger as _WandbLogger
from hive.utils.registry import registry
import wandb


class WandbLogger(_WandbLogger):
    """A Wandb logger.

    This logger can be used to log to wandb. It assumes that wandb is configured
    locally on your system. Multiple timescales/loggers can be implemented by
    instantiating multiple loggers with different logger_names. These should still
    have the same project and run names.

    Check the wandb documentation for more details on the parameters.
    """

    def __init__(
        self,
        timescales=None,
        logger_schedules=None,
        project=None,
        name=None,
        dir=None,
        mode=None,
        id=None,
        resume=None,
        start_method=None,
        **kwargs,
    ):
        super().__init__(
            timescales,
            logger_schedules,
            project,
            name,
            dir,
            mode,
            id,
            resume,
            start_method,
            **kwargs,
        )
        self._logged_timescales = set()

    def register_timescale(self, timescale, schedule=None, log_timescale=False):
        if log_timescale:
            self._logged_timescales.add(timescale)
        return super().register_timescale(timescale, schedule)

    def log_config(self, config):
        # Convert list parameters to nested dictionary
        for k, v in config.items():
            if isinstance(v, list):
                config[k] = {}
                for idx, param in enumerate(v):
                    config[k][idx] = param

        wandb.config.update(config)

    def log_scalar(self, name, value, prefix, step=None):
        metrics = {f"{prefix}/{name}": value}
        if step is not None:
            metrics.update({f"{step}_step": self._steps[step]})
        metrics.update(
            {
                f"{timescale}_step": self._steps[timescale]
                for timescale in self._logged_timescales
            }
        )
        wandb.log(metrics)

    def log_metrics(self, metrics, prefix, step=None):
        if prefix != "":
            prefix += "/"
        elif prefix is None:
            prefix = ""
        metrics = {f"{prefix}{name}": value for (name, value) in metrics.items()}
        if step is not None:
            metrics.update({f"{step}_step": self._steps[step]})
        metrics.update(
            {
                f"{timescale}_step": self._steps[timescale]
                for timescale in self._logged_timescales
            }
        )
        wandb.log(metrics)

    def finish(self):
        wandb._sentry.end_session()


def composite_logger_hack(loggers):
    return loggers[0]


registry.register("WandbLogger", WandbLogger, WandbLogger)
