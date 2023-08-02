# IBCL Code

### 0. Prerequisites

To run the project, please install Python >= 3.8 and the packages in the requirements.txt file by running

```
pip install -r requirements.txt
```

### 1. Preparing data

To preprocess data for both Split CIFAR-10 and CelebA, run the following command.

```
python ibcl_main/preprocess.py --task_name=<cifar10|celeba> --raw_data_dir=<your raw data dir> --proc_data_dir=<your proc data dir>
```

This will first download raw CIFAR10 or CelebA data from their torchvision sources.
The downloaded raw data will be saved in a directory, and processed data will be stored in another.
The preprocessing include transform and feature extraction by a pre-trained ResNet18.
Notice that CelebA may be too large to be downloaded from torchvision directly. In this case, please refer to [this link](https://github.com/pytorch/vision/issues/2262) to manually download it from Google drive,
and set `download=False` for all 3 datasets in `download_celeba` in `utils/preprocess_utils.py`.

### 2. Training FGCS

To update FGCS knowledge base by training, run the following command.

```
python ibcl_main/fgcs_update.py --task_name=<cifar10|celeba> --data_dir=<your proc data dir> --model_size=<small|normal|large>
```

We provide 3 different models (3 parameter spaces) for training. Their architectures are available in `models/models.py`. The default model is small.
For every task, this code will update the FGCS in `fgcs.pth` in the directory provided. FGCS across all tasks will be checkpointed in this file.
We also save a log of loss and a log of validation accuracy per epoch during training.

### 3. Zero-shot preference addressing

To locate model HDRs that address particular preferences, run the following command.

```
python ibcl_main/zero_shot_model_locate.py --task_name=<cifar10|celeba> --data_dir=<your proc data dir> --alpha=<a number between 0 and 1> --num_prefs_per_task=<number of preferences per task> --num_models_per_pref=<number of sampled models per preference>
```

This code will uniformly sample a number of preferences per task, except for the first task, which can only have preference = [1]. Then, for each preference, it computes an HDR and samples a number of models from the HDR.
It then evaluates the testing accuracy on all tasks encountered so far. The evaluated accuracy and sampled preferences will be saved in two dictionaries in the provided directory.
These results can be therefore used to compute metrics such as preference-weighted accuracy, average per-task accuracy, peak per-task accuracy and backward transfer.
Notice that we also provide a method `compute_pareto_front_two_tasks` to estimate the Pareto set of the first two tasks of a benchmark. This result can be visualized as Figure 2 and 3 in our paper.