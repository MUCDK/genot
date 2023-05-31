#!/usr/bin/env bash


set -ex

if ! python -c "import ott" 2>/dev/null ; then  # test to run once.
  # general
  apt-get update
  apt-get install -y libsndfile1-dev clang-format
  apt-get purge -y --allow-change-held-packages libnccl2 libnccl-dev

  # nccl 2.12.12
  pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  pip install ott-jax
  pip install jaxopt
  pip install optax

  # utils 
  pip install ml_collections
  pip install jax-dataloader
  pip install scikit-learn
  pip install scipy
  pip install seaborn
  pip install matplotlib
  pip install cloudpickle
  pip install h5py

  pip install . # TODO: check, does it work?


  # bolt 
  pip install --upgrade turibolt --index https://pypi.apple.com/simple

  # turitrove
  apt-get install libfuse-dev fuse -y
  apt-get install -y libblas3 liblapack3 libstdc++6 python-setuptools
  pip install --upgrade pip
  pip install turitrove -i https://pypi.apple.com/simple --upgrade

  # mount data
  mkdir datasets
  trove download dataset/shoes_64@1.0.0 datasets
  trove download dataset/handbags_64@1.0.0 datasets

  apt-get install unzip
  unzip /mnt/task_runtime/datasets/shoes_64-1.0.0/data/raw.zip
  unzip /mnt/task_runtime/datasets/handbags_64-1.0.0/data/raw.zip
fi
