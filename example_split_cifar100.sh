#!/bin/bash
pip install -r requirements.txt
python preprocess_data.py --task_name=cifar100 --raw_data_dir=raw_data --proc_data_dir=cifar100_proc_data
python fgcs_update.py --task_name=cifar100 --data_dir=cifar100_proc_data --model_size=small --discard_threshold=0.01
python generate_preferences.py --task_name=cifar100 --data_dir=cifar100_proc_data --num_prefs_per_task=10
python zero_shot_model_locate.py --task_name=cifar100 --data_dir=cifar100_proc_data --model_size=small --discard_threshold=0.01 --alpha=0.01 --num_models_per_pref=10
python visualize_results.py --task_name=cifar100 --data_dir=cifar100_proc_data --alpha=0.01 --discard_threshold=0.01