import argparse


def main(config: argparse.Namespace):
    import os
    import random
    from typing import Dict

    import cloudpickle

    # The names package was made available by installing it in the setup command.
    import h5py
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    # The turibolt package is available on all Bolt tasks.
    import turibolt as bolt

    from entot.data.data import create_gaussian_split, create_gaussians
    from entot.models.models import KantorovichGapModel, NoiseOutsourcingModel
    from entot.models.utils import DataLoader
    from entot.nets import UNet

    # helper functions (normally in jupyter notebook)
    def h5py_to_dataset(path, img_size=64, batch_size: int = 64):
        with h5py.File(path, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]

            # Get the data
            data = list(f[a_group_key])

            # dataset = 2 * jnp.transpose(jnp.array(data) / 255., (0, 3, 1, 2))- 1
            dataset = jnp.array(data) / 255.0
            dataset = jax.image.resize(dataset, (len(dataset), img_size, img_size, 3), method="bilinear")

        return DataLoader(dataset, batch_size=batch_size)

    def bolt_callback(progress: int, metrics: Dict[str, float]):
        bolt.send_metrics(
            {
                "Progress": progress,
                "Total loss": metrics["total_loss"][-1],
                "Fitting loss": metrics["fitting_loss"][-1],
                "Kant gap": metrics["kant_gap"][-1],
            }
        )

    # load data
    handbags = h5py_to_dataset(os.path.join(config.root_dir, "handbags_64-1.0.0/data/handbag_128.hdf5"))
    shoes = h5py_to_dataset(os.path.join(config.root_dir, "shoes_64-1.0.0/data/shoes_128.hdf5"))

    # set up model
    unet = UNet(diff_input_output=1)
    state_unet = unet.create_train_state(jax.random.PRNGKey(0), optax.adam(1e-4), (64, 64, 3))
    kg = KantorovichGapModel(epsilon_kant_gap=0.1, input_dim=[64, 64, 3], noise_dim=1, iterations=5000, neural_net=unet)

    # start training
    kg(handbags, shoes, callback=bolt_callback)

    with open(os.path.join(config.save_dir, "kant_neural_net.pkl"), "wb") as f:
        cloudpickle.dump(kg.neural_net)
    with open(os.path.join(config.save_dir, "kant_model.pkl"), "wb") as f:
        cloudpickle.dump(kg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default="./")

    main(parser.parse_args())
