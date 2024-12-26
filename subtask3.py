import os
from os.path import join, splitext
from typing import Final

import train
config_dir: Final[str] = "configs/subtask3/"
output_dir: Final[str] = "Graphs/subtask3/"
result_npz_dir: Final[str] = "Results/subtask3/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(result_npz_dir, exist_ok=True)
for file in os.listdir(config_dir):
    train.main(
        join(config_dir, file),
        join(output_dir, splitext(file)[0] + ".png"),
        join(result_npz_dir, splitext(file)[0] + ".npz")
    )