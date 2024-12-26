import os
from os.path import join, splitext
import shutil
from typing import Final

import train
config_dir: Final[str] = "configs/subtask3/"
finish_config_dir: Final[str] = "configs/subtask3_finish/"
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
    shutil.move(join(config_dir, file), join(finish_config_dir, file))