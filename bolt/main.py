import os
import sys
import shutil

import jax
from absl import app, flags
from ml_collections import config_flags


sys.path.append("/mnt/task_runtime/experiments")  # to get the import that works
# needs to have `main()` method which accepts the config
import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string("datadir", None, "Directory to load data from.")
flags.DEFINE_boolean("evaluate", False, "Predict loss per point.")
config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)
flags.mark_flags_as_required(["workdir"])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # prevent tf from allocating accelerator mem.

    # save hyperparameter config.
    config_file = flags.FLAGS["config"].config_filename
    os.makedirs(FLAGS.workdir, exist_ok=True)
    try:
        shutil.copy(config_file, os.path.join(FLAGS.workdir, "config.py"))
    except shutil.SameFileError:
        pass

    config = FLAGS.config
    run_experiment.main(config)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
