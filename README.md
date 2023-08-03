# IBCL Code

## Update 08/02/2023

First, we debugged our code to make sure it is runnable. This includes:

1. Fixed module import errors by modifying the directory structure.

2. Fixed argument errors in `general_model` in `fgcs_update.py`.

3. Fixed type error of input arguments to `zero_shot_model_locate.py`.

4. Modified the instructions below accordingly.

Next, we added a visualization script for the final results. Specifically,

5. Added Python script `visualize_results.py` and modified the instructions accordingly.


## Instructions of running our code

### 0. Prerequisites

To run the project, please install Python >= 3.8 and the packages in the requirements.txt file by running

```
pip install -r requirements.txt
```

### 1. Preparing data

To preprocess data for both Split CIFAR-10 and CelebA, run the following command.

```
python preprocess_data.py --task_name=<cifar10|celeba> --raw_data_dir=<path to your raw data dir> --proc_data_dir=<path to your proc data dir>
```

Notice that it is not necessary to creat these two directories beforehand, because this command will automatically create them, as long as the paths are valid.
This will first download raw CIFAR10 or CelebA data from their torchvision sources.
The downloaded raw data will be saved in a directory, and processed data will be stored in another.
The preprocessing include transform and feature extraction by a pre-trained ResNet18.
Notice that CelebA may be too large to be downloaded from torchvision directly. In this case, please refer to [this link](https://github.com/pytorch/vision/issues/2262) to manually download it from Google drive,
and set `download=False` for all 3 datasets in `download_celeba` in `utils/preprocess_utils.py`.

### 2. Training FGCS

To update FGCS knowledge base by training, run the following command.

```
python fgcs_update.py --task_name=<cifar10|celeba> --data_dir=<path to your proc data dir> --model_size=<small|normal|large>
```

We provide 3 different models (3 parameter spaces) for training. Their architectures are available in `models/models.py`. The default model is small.
For every task, this code will update the FGCS in `fgcs.pth` in the preprocessed data directory provided. FGCS across all tasks will be checkpointed in this file.
We also save a log of loss and a log of validation accuracy per epoch during training.

### 3. Zero-shot preference addressing

To locate model HDRs that address particular preferences, run the following command.

```
python zero_shot_model_locate.py --task_name=<cifar10|celeba> --data_dir=<your proc data dir> --alpha=<a number between 0 and 1> --num_prefs_per_task=<number of preferences per task> --num_models_per_pref=<number of sampled models per preference>
```

This code will uniformly sample a number of preferences per task, except for the first task, which can only have preference = [1]. Then, for each preference, it computes an HDR and samples a number of models from the HDR.
It then evaluates the testing accuracy on all tasks encountered so far. The evaluated accuracy and sampled preferences will be saved in two dictionaries in the provided directory.
These results can be therefore used to compute metrics such as preference-weighted accuracy, average per-task accuracy, peak per-task accuracy and backward transfer.
Notice that we also provide a method `compute_pareto_front_two_tasks` to estimate the Pareto set of the first two tasks of a benchmark. This result can be visualized as Figure 2 and 3 in our paper.

### 4. Visualize the results

![alt](figs/cifar10_avg_acc_example.png =250x) ![alt](figs/cifar10_peak_acc_example.png =250x) ![alt](figs/cifar10_avg_bt_example.png =250x)

We added a script to help visualize the continual learning metrics as we did in Figure 7 of Appendix I. This includes average per task accuracy,
peak per task accuracy and average per task backward transfer. To do so, run the following command.

```
python visualize_results.py --task_name=<cifar10|celeba> --data_dir=<your proc data dir> --alpha=<a number between 0 and 1>
```

This will fetch the accuracy pt file saved from last step and produce 3 figures similar to the ones above.
Notice that this code does not visualize baseline results for comparison yet, but we can add this function per request.


## Example Split-CIFAR10 bash script

We have included an example run of the entire three steps on Split-CIFAR10 as a bash script `example_split_cifar10.sh`.
This script starts from preprocessing the data and ends up visualizing the results.
If calling this bash script does not work, please refer to the step-by-step instructions above to produce results.