# logging_setup.py
import logging, os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    Configure un logger applicatif standard :
      - Console (stream) + fichier en rotation quotidienne (logs/app.log)
      - Niveau par défaut INFO (paramétrable)
      - Format uniforme avec timestamp et module
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    if logger.handlers:
        # Déjà configuré → on ne double pas les handlers
        return

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Fichier (rotation quotidienne, conserve 14 jours)
    fh = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        when="D",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        delay=False
    )
    fh.setLevel(logger.level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
