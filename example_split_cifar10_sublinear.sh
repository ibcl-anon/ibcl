#!/bin/bash
pip install -r requirements.txt
python preprocess_data.py --task_name=cifar10 --raw_data_dir=raw_data --proc_data_dir=proc_data
python fgcs_update_sublinear.py --task_name=cifar10 --data_dir=proc_data --model_size=small --discard_threshold=0.01
python zero_shot_model_locate_sublinear.py --task_name=cifar10 --data_dir=proc_data --model_size=small --alpha=0.75 --num_prefs_per_task=10 --num_models_per_pref=10
python visualize_results.py --task_name=cifar10 --data_dir=proc_data --alpha=0.75 --sublinear=1