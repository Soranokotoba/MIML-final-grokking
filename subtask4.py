import os
from os.path import join, splitext
import shutil
from typing import Final

config_path: Final[str] = "configs/subtask4/subtask4.yaml"
output_dir: Final[str] = "Graphs/subtask4/"
result_npz_dir: Final[str] = "Results/subtask4/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(result_npz_dir, exist_ok=True)
import train
train.main(config_path, join(output_dir, "subtask4.png"), join(result_npz_dir, "subtask4.npz"))