"""Default Hyperparameter configuration."""

import ml_collections


def bolt_config():
    config = ml_collections.ConfigDict()
    config.tags = ["ot"]
    config.command = "bash bolt/run_experiment.sh"
    config.is_parent = False
    config.priority = 1
    config.project_id = "mlr_understanding"


    config.resources = ml_collections.ConfigDict()
    res = config.resources
    res.docker_image = "docker.apple.com/tata-antares/mlx-cuda-11.8-cudnn8-ubuntu20.04-nccl-2.17.1-efa-1.22.1"
    res.cluster = "gcp1"
    res.num_gpus = 1
    res.ports = ["TENSORBOARD_PORT", "JUPYTER_PORT"]
    res.timeout = "14d"
    res.memory_gb = 128
    res.disk_gb = 128

    config.environment_variables = ml_collections.ConfigDict()
    env = config.environment_variables
    env.CUDA_HOME = "/usr/local/cuda-11.4"  # we enable cuda-compat libs here
    env.LD_LIBRARY_PATH = "/usr/local/cuda-11.4/compat:$LD_LIBRARY_PATH"
    env.TF_CUDNN_DETERMINISTIC = 1
    env.INIT_CHECKPOINT = ""

    config.permissions = ml_collections.ConfigDict()
    config.permissions.viewers = ["mlr"]
    return config


def make_one_config() -> ml_collections.ConfigDict:
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.bolt = bolt_config()

    config.bolt.name = f"Making config"

    config.save_path = "/mnt/task_wrapper/user_output/artifacts"
    config.root_dir = "/mnt/task_runtime/datasets"

    return config


def get_config():
    configs = []
    configs.append(make_one_config())
    if len(configs) > 1:
        config = ml_collections.ConfigDict()
        for i, c in enumerate(configs):
            config["config_%d" % i] = c
    else:
        config = configs[0]
    return config
