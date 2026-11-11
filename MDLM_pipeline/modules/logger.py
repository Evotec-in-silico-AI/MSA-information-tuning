import logging

logger = logging.getLogger("protein-experiments")


def configure_default_logging() -> None:
    """Configure default logging if no handlers have been installed."""
    if logger.handlers:
        return
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d--%H:%M:%S",
    )


configure_default_logging()
