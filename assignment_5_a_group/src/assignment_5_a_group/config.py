"""This module contains the general configuration of the project."""
from pathlib import Path


SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

GROUPS = ["marital_status", "qualification"]

cat = ["antisocial", "anxiety", "headstrong", "hyperactive", "peer"]

bpi_cat = ["bpiA", "bpiB", "bpiC", "bpiD", "bpiE"]

__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]
