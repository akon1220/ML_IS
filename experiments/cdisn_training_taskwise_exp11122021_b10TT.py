# cdisn_training_taskwise_exp11122021.py: a file defining an experiment to run a taswise CDISN ensemble on CPNE's Fluoxetine data
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import os
import argparse
import pickle as pkl
from itertools import (
    chain,
    combinations,
)  # see https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset

from train.train_taskwise_cdisn_ensemble import train_cdisn_ensemble_taskwise
from utils.caching_utils import create_directory


def kick_off_training_run(alg_args, curr_run_meta_params):
    """
    cdisn_training_taskwise_exp11122021.kick_off_training_run: kicks off a job to train a cdisn ensemble in a taskwise fashion
    - Inputs:
        * alg_args (): ,
        * curr_run_meta_params (): ,
    - Outputs:
        * N/A
    - Usage:
        * N/A

    #########
    Note: final alg_args for the job should resemble:
        save_dir,
        root_data_dirs,  # list of datasets for each task (ordered according to requested_tasks)
        train_set_foldernames,
        val_set_foldernames,
        test_set_foldernames,
        requested_tasks=["RP", "TS", "BehavioralTUAB"], #, "BehavioralFluoxetine"],
        cv_split_nums=[0, 0, 0], #, 0],
        max_num_retrain_iterations=10,
        batch_size=50,
        shuffle=True,
        max_epochs=100,
        max_evals_after_saving=6,
        save_freq=20,
        models_root_file_name="RP_TS_BehavioralTUAB_Training, # "RP_with_BehavioralFluoxetine_Training",
        dropout_rate=0.5,
        embed_dim=100,
        num_channels=21,
        window_len=600,
        num_classes=[
            2,
            2,
            2, # 4,
        ],  # list of classes for each task (ordered according to requested_tasks)
        beta_vals=(0.9, 0.999),
        learning_rate=2e-4,
        weight_decay=0.001,
        former_state_dict_file=None,
        pretrained_upstream_cdisn_file_path=None,
    """

    print("STARTING RUN WITH META PARAMS == ", curr_run_meta_params)

    # define indices into alg_args that need to be populated
    SAVE_DIR_INDEX = 0
    ROOT_DATA_DIRS_INDEX = 1
    TRAIN_SET_FOLDERS_INDEX = 2
    VAL_SET_FOLDERS_INDEX = 3
    TEST_SET_FOLDERS_INDEX = 4
    REQUESTED_TASKS_INDEX = 5
    CV_SPLIT_NUMS_INDEX = 6
    MODELS_ROOT_FILE_NAME_INDEX = 13
    # NUM_CHANS_INDEX = 16
    # WINDOW_LEN_INDEX = 17
    NUM_CLASSES_INDEX = 18

    # obtain values for arguments in curr_run_meta_params that need to be placed into alg_args
    save_dir = curr_run_meta_params[0]
    root_data_dirs = curr_run_meta_params[1]
    train_set_folders = curr_run_meta_params[2]
    val_set_folders = curr_run_meta_params[3]
    test_set_folders = curr_run_meta_params[4]
    requested_tasks = curr_run_meta_params[5]
    cv_split_nums = curr_run_meta_params[6]
    models_root_file_name = curr_run_meta_params[7]
    # num_channels = curr_run_meta_params[8]
    # window_len = curr_run_meta_params[9]
    num_classes = curr_run_meta_params[8]

    # update alg_args
    alg_args[SAVE_DIR_INDEX] = save_dir
    alg_args[ROOT_DATA_DIRS_INDEX] = root_data_dirs
    alg_args[TRAIN_SET_FOLDERS_INDEX] = train_set_folders
    alg_args[VAL_SET_FOLDERS_INDEX] = val_set_folders
    alg_args[TEST_SET_FOLDERS_INDEX] = test_set_folders
    alg_args[REQUESTED_TASKS_INDEX] = requested_tasks
    alg_args[CV_SPLIT_NUMS_INDEX] = cv_split_nums
    alg_args[MODELS_ROOT_FILE_NAME_INDEX] = models_root_file_name
    # alg_args[NUM_CHANS_INDEX] = num_channels
    # alg_args[WINDOW_LEN_INDEX] = window_len
    alg_args[NUM_CLASSES_INDEX] = num_classes

    # create the new directory for saving info to
    # see https://www.geeksforgeeks.org/python-os-path-exists-method/ and https://www.geeksforgeeks.org/python-os-mkdir-method/
    if not os.path.exists(save_dir):
        create_directory(save_dir)

    # train the upstream model
    train_cdisn_ensemble_taskwise(*alg_args)

    print("DONE RUNNING EXPERIMENT WITH META PARAMS == ", curr_run_meta_params)
    pass


def build_list_of_split_combos(available_split_sets_by_task, split_combos):
    if len(available_split_sets_by_task) == 0:
        return split_combos
    else:
        new_split = available_split_sets_by_task[0]
        if len(split_combos) == 0:
            for fold in new_split:
                split_combos.append([fold])
        else:
            new_split_combos = []
            for i, combo in enumerate(split_combos):
                for fold in new_split:
                    new_split_combos.append(combo + [fold])
            split_combos = new_split_combos
        return build_list_of_split_combos(
            available_split_sets_by_task[1:], split_combos
        )


def set_up_and_run_current_experiment(args):
    """
    cdisn_training_taskwise_exp11122021.set_up_and_run_current_experiment: code for parallelizing and calling cdisn training jobs
    - Inputs:
        * alg_args (): ,
    - Outputs:
        * N/A
    - Usage:
        * N/A
    """
    root_save_dir = args.root_save_dir
    all_supported_tasks = args.all_supported_tasks
    all_supported_tasks_classmap = args.all_supported_tasks_classmap
    all_supported_datasets = args.all_supported_datasets
    for i, task in enumerate(
        all_supported_tasks
    ):  # ensure datasets are matched to supported_tasks
        assert task in all_supported_datasets[i]
    full_parameter_dict = args.full_parameter_dict

    # print("cdisn_training_taskwise_exp11122021.set_up_and_run_current_experiment: WARNING - CURRENT IMPLEMENTATION ASSUMES ALL DATASETS SHOULD HAVE SAME NUMBER OF CHANNELS")
    # good_channels_across_datasets = {}
    # num_channels = None
    # window_len = None
    # for i, dataset_path in enumerate(all_supported_datasets):
    #     with open(dataset_path+os.sep+"cached_data_stats_and_params.pkl", 'rb') as infile:
    #         curr_cached_data_stats_and_params = pkl.load(infile)
    #         curr_good_channels = curr_cached_data_stats_and_params['good_channels']
    #         curr_window_len = curr_cached_data_stats_and_params.task_variable_parameter_args.window_len
    #         if i > 0: # ensure that all datasets use the same channels
    #             assert len(curr_good_channels) == len(good_channels_across_datasets[-1])
    #             for c1, c2 in zip(sorted(curr_good_channels), sorted(good_channels_across_datasets[-1])):
    #                 assert c1 == c2
    #             assert curr_window_len == window_len
    #         else:
    #             num_channels = len(curr_good_channels)
    #             window_len = curr_window_len
    #         good_channels_across_datasets.append(curr_good_channels)

    # see https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    tasks_to_parallelize = list(
        chain.from_iterable(
            combinations(all_supported_tasks, r)
            for r in range(len(all_supported_tasks) + 1)
        )
    )[1:]
    data_for_tasks = list(
        chain.from_iterable(
            combinations(all_supported_datasets, r)
            for r in range(len(all_supported_datasets) + 1)
        )
    )[1:]

    all_parameter_settings = []
    for i, task_hyperparams in enumerate(zip(tasks_to_parallelize, data_for_tasks)):
        requested_tasks = list(task_hyperparams[0])
        main_datasets_for_requested_tasks = list(task_hyperparams[1])

        requested_data_splits_to_parallelize = []
        for dataset_path in list(main_datasets_for_requested_tasks):
            train_sets = sorted([x for x in os.listdir(dataset_path) if "train" in x])
            val_sets = sorted([x for x in os.listdir(dataset_path) if "val" in x])
            test_sets = sorted([x for x in os.listdir(dataset_path) if "test" in x])
            if len(test_sets) == 0:
                test_sets = [None]

            assert len(train_sets) == len(val_sets)
            assert len(test_sets) == 1

            curr_dataset_split_combos = []
            for ind in range(len(train_sets)):
                curr_dataset_split_combos.append(
                    [train_sets[ind], val_sets[ind], test_sets[0]]
                )
                # curr_dataset_split_combos.append([dataset_path+os.sep+train_sets[ind], dataset_path+os.sep+train_sets[ind], test_sets[0]])
            requested_data_splits_to_parallelize.append(curr_dataset_split_combos)

        all_task_split_combinations = build_list_of_split_combos(
            requested_data_splits_to_parallelize, []
        )

        for split_combo in all_task_split_combinations:
            assert len(split_combo) == len(requested_tasks)
            assert len(split_combo[0]) == 3

            curr_supported_datasets = []
            for task_id in requested_tasks:
                for path in all_supported_datasets:
                    if "_"+task_id+"_" in path:
                        curr_supported_datasets.append(path)

            curr_train_sets = [x[0] for x in split_combo]
            curr_val_sets = [x[1] for x in split_combo]
            curr_test_sets = [x[2] for x in split_combo]
            cv_split_nums_by_task = [int(x[0][-1]) for x in split_combo]
            models_root_file_name = "_".join(
                [
                    task + str(cv_num)
                    for task, cv_num in zip(requested_tasks, cv_split_nums_by_task)
                ]
            )
            curr_save_dir = root_save_dir + os.sep + models_root_file_name
            num_classes = [
                all_supported_tasks_classmap[task] for task in requested_tasks
            ]

            all_parameter_settings.append(
                (
                    curr_save_dir,
                    curr_supported_datasets, # all_supported_datasets,
                    curr_train_sets,
                    curr_val_sets,
                    curr_test_sets,
                    requested_tasks,
                    cv_split_nums_by_task,
                    models_root_file_name, # num_channels, window_len,  
                    num_classes, 
                )
            )

    # run the parallelized jobs
    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print("<<< KICKING OFF TASKID == ", taskID)
    task_param_settings = all_parameter_settings[taskID - 1]
    kick_off_training_run(full_parameter_dict["cdisn"], task_param_settings)
    return taskID


if __name__ == "__main__":
    """
    cached_args.pkl arguments:
        args.root_save_dir = str
        args.all_supported_tasks = ['RP', 'TS', 'BehavioralTUAB'] #, 'BehavioralFluoxetine']
        args.all_supported_tasks_classmap = {'RP':2, 'TS':2, 'BehavioralTUAB':2} #, 'BehavioralFluoxetine':4}
        args.all_supported_datasets = [rp_data_path, ts_data_path, BehavioralTUAB_data_path] #, behavioral_fluox_path]
        args.full_parameter_dict: {'cdisn':[*args_for_training_loop]}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cached_args_file",
        type=str,
        default="cached_args_cdisn_training_taskwise_exp11122021_b10TT.pkl",
        help="/path/to/cached_args.pkl should contain an argparse.ArgumentParser() with arguments structured as in the __main__ doc string",
    )
    parser.add_argument(
        "-n", default=None, help="IGNORE THIS ARGUMENT - IT'S A BANDAID"
    )

    args = parser.parse_args()

    print("<<< MAIN: START")
    with open(args.cached_args_file, "rb") as infile:
        cached_args = pkl.load(infile)
        print("<<< cached_args == ", cached_args)
        args.root_save_dir = cached_args.root_save_dir
        args.all_supported_tasks = cached_args.all_supported_tasks
        args.all_supported_tasks_classmap = cached_args.all_supported_tasks_classmap
        args.all_supported_datasets = cached_args.all_supported_datasets
        args.full_parameter_dict = cached_args.full_parameter_dict
        print("<<< new args == ", args)

    taskID = set_up_and_run_current_experiment(args)

    print("<<< MAIN: DONE RUNNING TASKID == ", taskID, "!!!")
    pass

