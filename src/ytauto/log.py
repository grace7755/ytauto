import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler


def setup(project_dir: Path | None = None, verbose: bool = False) -> logging.Logger:
    lg = logging.getLogger("ytauto")
    lg.handlers.clear()
    lg.setLevel(logging.DEBUG)
    ch = RichHandler(show_path=False, rich_tracebacks=True, markup=False)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    lg.addHandler(ch)
    if project_dir:
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(project_dir / "logs" / "run.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


log = logging.getLogger("ytauto")
