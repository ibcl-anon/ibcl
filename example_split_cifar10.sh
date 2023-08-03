#!/bin/bash
pip install -r requirements.txt
python preprocess_data.py --task_name=cifar10 --raw_data_dir=raw_data --proc_data_dir=proc_data
python fgcs_update.py --task_name=cifar10 --data_dir=proc_data --model_size=small
python zero_shot_model_locate.py --task_name=cifar10 --data_dir=proc_data --alpha=0.75 --num_prefs_per_task=10 --num_models_per_pref=10
python visualize_results.py --task_name=cifar10 --data_dir=proc_data --alpha=0.75