import os

# Figures are written to disk, so no display is needed.
os.environ.setdefault("MPLBACKEND", "Agg")

from .pipeline import run

if __name__ == "__main__":
    run()
