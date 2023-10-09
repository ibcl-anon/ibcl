# IBCL Code

This is the anonymous GitHub repo of paper "IBCL: Zero-shot Model Generation for Task Trade-offs in Continual Learning".

![IBCL Workflow(figs/ibcl_flowchart.png)


## Instructions of running our code

### 0. Prerequisites

To run the project, please install Python >= 3.8 and the packages in the requirements.txt file by running

```
pip install -r requirements.txt
```

### 1. Preparing data

There are 5 valid benchmarks supported by IBCL so far. The valid tasks names are `cifar10`, `celeba`, `cifar100`, `tinyimagenet`, `20newsgroups`.
- For CelebA, our auto download interface may not work due to the file size is large. In this case, please refer to [this link](https://github.com/pytorch/vision/issues/2262) to manually download it from Google drive to your raw data directory
and set `download=False` for all 3 datasets in `download_celeba` in `utils/preprocess_utils.py`.
- For TinyImageNet, data needs to be first manually downloaded to the raw data directory by [this link](http://cs231n.stanford.edu/tiny-imagenet-200.zip).

After the manual download is done for CelebA or TinyImageNet, run the following command. (Nothing needs to be done before running this command if it is another benchmark.)

```
python preprocess_data.py --task_name=<a valid task name> --raw_data_dir=<path to your raw data dir> --proc_data_dir=<path to your proc data dir>
```

Notice that it is not necessary to creat these two directories beforehand, because this command will automatically create them, as long as the paths are valid.
The downloaded raw data will be saved in the raw data directory, and processed data will be stored in the other.


### 2. Training FGCS

To update FGCS knowledge base by training as in Algorithm 1, run the following command.

```
python fgcs_update.py --task_name=<a valid task name> --data_dir=<path to your proc data dir> --model_size=<small|normal|large> --discard_threshold=<a nonnegative floating point>
```

We provide 3 different models (3 parameter spaces) for training. Their architectures are available in `models/models.py`. The default model is small.
The argument `discard_threshold` is the hyperparameter $d$ in our paper. The default value is 0.01.
For every task, this code will update the FGCS in `fgcs.pth` in the preprocessed data directory provided. FGCS across all tasks will be checkpointed in this file.
The reuse map of learned posteriors is stored in `dict_reuse_map_{discard_threshold}.pt`.
The buffer growth log is stored in `sublinear_buffer_growth_{discard_threshold}.pt`.
We also save a log of loss and a log of validation accuracy per epoch during training.


### 3. Randomly generate task preferences

This step can be done before step 2. Run the following command to generate preferences over the tasks for a given benchmark.

```
python generate_preferences.py --task_name=<a valid task name> --data_dir=<path to your proc data dir> --num_prefs_per_task=<number of preferences per task>
```

The default number of preferences at each task is 10. The generated preferences will be saved in `dict_prefs.pt` under your processed data directory.

### 4. Zero-shot preference addressing

To locate model HDRs that address particular preferences as in Algorithm 2, run the following command.

```
python zero_shot_model_locate.py --task_name=<a valid task name> --data_dir=<your proc data dir> --model_size=<small|normal|large> --discard_threshold=<the same discard threshold in step 2> --alpha=<a number between 0 and 1> --num_models_per_pref=<number of sampled models per preference>
```

For each preference, this code computes an HDR and samples a number of models from the HDR.
It then evaluates the testing accuracy on all tasks encountered so far. The evaluated accuracy and sampled preferences will be saved in two dictionaries in the provided directory.
These results can be therefore used to compute metrics such as preference-weighted accuracy, average per-task accuracy, peak per-task accuracy and backward transfer.
The default value of `alpha` is 0.01.
An accuracy matrix will be generated in `dict_all_accs_{alpha}_{discard_threshold}.pt`.

### 5 Results visualization

We added a script to help visualize the continual learning metrics. This includes computing average per task accuracy,
peak per task accuracy and average per task backward transfer from `dict_all_accs_{alpha}_{discard_threshold}.pt` and plot the outcomes. To do so, run the following command.

```
python visualize_results.py --task_name=<a valid task name> --data_dir=<your proc data dir> --alpha=<the same alpha in step 4> --discard_threshold=<the same discard threshold in step 4>
```

This will fetch the accuracy `.pt` file saved from last step and produce 3 figures, one for each metric.
Notice that this code does not visualize baseline results for comparison yet. To do so, refer to step 6 and follows.


### 6. An example Split-CIFAR100 bash script

We have included an example run of the entire three steps on Split-CIFAR100 as a bash script `example_split_cifar100.sh`.
This script starts from preprocessing the data and ends up visualizing the results.
If calling this bash script does not work, please refer to the step-by-step instructions above to produce results.


## Instructions of running baseline methods and visualize comparative results

### 7. Running GEM and VCL

After we have finished preprocessing data and generate preferences (i.e. after step 1 and 3), we can call GEM and VCL to generate baseline results as in our experiments.
This can be done by the following commands.

```
python baseline_gem.py --task_name=<a valid task name> --data_dir=<your proc data dir> --model_size=<small|normal|large>
```

```
python baseline_vcl.py --task_name=<a valid task name> --data_dir=<your proc data dir> --model_size=<small|normal|large> --num_models_per_pref=<number of sampled models per preference>
```

Notice that GEM is a deterministic machine learning algorithm, so it does not sample models after producing a probabilistic solution. 
That is, there is no `num_models_per_pref`. Running `baseline_gem_reg.py` is similar to `baseline_gem.py` and
running `baseline_vcl_reg.py` is similar to running `baseline_vcl.py`.


### 8. Visualizing GEM, VCL and IBCL results together

After we obtain results from IBCL and the baselines, we can visualize comparative results on average per task accuracy,
peak per task accuracy and backward transfer, as we did in Section 5.2.

```
python visualize_results_w_baselines.py --task_name=<ca valid task name> --data_dir=<your proc data dir> --alpha=<the same alpha in step 4> --discard_threshold=<the same discard threshold in step 4>
```