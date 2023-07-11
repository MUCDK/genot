import os
from code import interact

import turibolt as bolt
from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("tar", ".", "Code directory.")
flags.DEFINE_bool("interactive", False, "Interactive mode.")
config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def with_indent(text, indent=4):
    return "".join([" " * indent + t for t in text.splitlines(True)])


def launch(config, tar, interactive):
    print("Launch %s" % config["bolt"]["name"])
    filename = os.path.join(tar, "config_for_bolt_machine.py")
    py_code = "import ml_collections\n\n"
    py_code += "true = True\n"
    py_code += "null = None\n"
    py_code += "false = False\n\n"
    py_code += "def get_config():\n"
    py_code += with_indent("return ml_collections.ConfigDict(" + config.to_json(indent=4))
    py_code += ")\n"

    with open(filename, "w") as fin:
        fin.write(py_code)

    bolt.submit(config=config.bolt.to_dict(), tar=tar, interactive=interactive)
    os.remove(filename)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config = FLAGS.config
    interactive = FLAGS.interactive
    tar = FLAGS.tar

    if "config_0" in config:  # multiple experiment sweep
        assert not interactive
        i = 0
        while "config_%d" % i in config:
            launch(config["config_%d" % i], tar, interactive)
            i += 1
    else:  # single experiment
        launch(config, tar, interactive)


if __name__ == "__main__":
    app.run(main)
