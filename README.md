# Earthquake detection via fiber optic cables using deep learning

- **Author:** Fantine Huot

## Getting started

### Update the submodules
After cloning the repository, run the following commands to initialize and
update the submodules.

```bash
git submodule init
git submodule update
```

### Requirements

You can run the project from an interactive bash session within the provided
[Docker](https://www.docker.com]) container:
```bash
docker run --gpus all -it fantine/ml_framework:latest bash
```
If you do not have root permissions to run Docker, [Singularity](https://singularity.lbl.gov) might be a good alternative for you. Refer to 
`containers/README.md` for more details.


## Folder structure

- **bin:** Scripts to run machine learning jobs.
- **catalog:** Earthquake and background noise database. 
- **config:** Configuration files. 
- **containers:** Details on how to use containers for this project. 
- **das_reader:** Legacy code for reading SEGY files.
- **docs:** Documentation.
- **hptuning:** Hyperparameter tuning for machine learning.
- **log:** Directory for log files.
- **ml_framework:** Machine learning framework.
- **preprocessing:** Data preprocessing steps.
- **processing_utils:** Processing utility functions.
- **tfrecords:** Utility functions for converting input files to TFRecords.

## Set the datapath for the project

Set the `DATAPATH` variable inside `config/datapath.sh` to the data or scratch directory
to which you want write data files.

## Create and run a machine learning model

This repository provides a parameterized, modular framework for creating and
running ML models.

- [Convert input data to TensorFlow records](docs/convert_tfrecords.md)
- [Machine learning training and inference](docs/ml_framework.md)
- [Hyperparameter tuning](docs/hptuning.md)
