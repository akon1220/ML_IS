# preprocess_CPNEFluoxetine_data.py: a file defining CPNE-Fluoxetine dataset-specific preprocessing classes and functions
# SEE LICENSE STATEMENT AT THE END OF THE FILE

# dependency import statements
import os
import pickle as pkl
import argparse
from random import shuffle, randint
import scipy.io as scio
import numpy as np
import mne

# from utils.caching_utils import create_directory
from data_preprocessing.data_preprocessing_utils import (
    PreprocessedDataset,
    randomly_sample_window_from_signal,
    sample_window_from_signal_for_RP_task,
    sample_window_from_signal_for_TS_task,
)


class CPNEFluoxetineDataset(PreprocessedDataset):
    """
    CPNEFluoxetineDataset Description:
    - General Purpose: defines a class for preprocessing and caching the CPNE Fluoxetine dataset
    - Usage:
      * preprocess_CPNEFluoxetine_data.__main__: uses this class to perform shuffling, splitting, and caching operations on a given CPNE Fluoxetine repository
    """

    def __init__(
        self,
        root_save_directory,
        data_split_type,
        task_id,
        data_source_id,
        date_of_data_split,
        task_variable_parameter_args=None,
        data_source_specific_args=None,
        num_splits=1,
        holdout_set=False,
        split_ratios=(0.7, 0.2, 0.1),
    ):
        """
        CPNEFluoxetineDataset.__init__: defines directory, filenames into which preprocessed data will be saved in addition to member variables
         - Inputs:
            * root_save_directory (str): the (root) directory to which the preprocessed data will be saved - see templates.file_directory_templates for details
            * data_split_type (str): one of ['standardSplit', 'crossValSplit'] denoting the type of dataset being curated
            * task_id (str): one of the id's present in self.get_hyperparams_for_task cases denoting the types of tasks being modeled in the current dataset
            * data_source_id (str): an identifier of the original dataset (e.g., 'TUAB' or 'CPNEFluoxetine')
            * date_of_data_split (str): a YYYYMMDD-formated string denoting the date of dataset curation/preprocessing
            * task_variable_parameter_args (argparse obj): arguments for task-specific parameters (e.g., tau-positive for RP task) which we may wish to tune in experiments
            * data_source_specific_args (argparse obj): arguments for preprocessing the CPNE Fluoxetine dataset, default=None
            * num_splits (int): the number of train-val-test splits to make (esp. for cross-validation), default=1
            * holdout_set (boolean): whether a test split needs to be stored as a holdout set, default=False
            * split_ratios (tuple): a tuple containing the ratios (float) of each split data subset to be assigned to the train, validation, and test sets, respectively, default=(0.7, 0.2, 0.1)
         - Outputs:
            * preprocessed-dataset (CPNEFluoxetineDataset): a fully preprocessed CPNE Fluoxetine dataset ready for consummption by the cross-domain-learning repository
         - Usage:
            * CPNEFluoxetineDataset class: uses CPNEFluoxetineDataset.__init__ to initialize member variables
        """
        super(CPNEFluoxetineDataset, self).__init__(
            root_save_directory,
            data_split_type,
            task_id,
            data_source_id,
            date_of_data_split,
            task_variable_parameter_args,
            data_source_specific_args,
            num_splits,
            holdout_set,
            split_ratios,
        )
        pass

    def get_class_info_for_data_file(self, data_file_path):
        """
        CPNEFluoxetineDataset.draw_sample: draws a random sample from a data file corresponding the self.task_id
         - Inputs:
            * data_file_path (str): the path to the source data file from which a sample is to be drawn
         - Outputs:
            * class_label (int): an int representing the behavioral class label corresponding to the data in data_file_path
         - Usage:
            * CPNEFluoxetineDataset.draw_sample: uses CPNEFluoxetineDataset.get_class_info_for_data_file to determine if
              data_file_path corresponds to saline or fluoxetine behavioral classes
        """
        SALINE_LABEL = 2
        FLUOXETINE_LABEL = 3

        split_file_path = data_file_path.split("_")
        unique_identifier_for_curr_data = "_".join(split_file_path[:2])
        data_file_class_info_dir = os.sep.join(split_file_path[:-2] + ["ClassData"])

        curr_data_class_info_file_names = [
            x
            for x in os.listdir(data_file_class_info_dir)
            if unique_identifier_for_curr_data in x
        ]
        assert len(curr_data_class_info_file_names) == 1
        curr_data_class_info_file_name = curr_data_class_info_file_names[0]

        curr_data_class_info = scio.loadmat(
            data_file_class_info_dir + os.sep + curr_data_class_info_file_name
        )
        if sum(curr_data_class_info["Fluoxetine"][0]) == 0:
            assert sum(curr_data_class_info["Saline"][0]) > 0
            return SALINE_LABEL
        elif sum(curr_data_class_info["Fluoxetine"][0]) > 0:
            assert sum(curr_data_class_info["Saline"][0]) == 0
            return FLUOXETINE_LABEL
        else:
            raise ValueError(
                "CPNEFluoxetineDataset.get_class_info_for_data_file: unexpected sum of class info"
            )

    def draw_sample(self, file_path):
        """
        CPNEFluoxetineDataset.draw_sample: draws a random sample from a data file corresponding the self.task_id
         - Inputs:
            * file_path (str): the path to the source data file from which a sample is to be drawn
         - Outputs:
            * sample (tuple): a tuple formatted as (x1, x2, ..., xn, label_y), representing a single data point for training on self.task_id
         - Usage:
            * CPNEFluoxetineDataset.cache_samples_for_current_split_fold: uses CPNEFluoxetineDataset.draw_sample prior to draw a single sample

        Note: this function draws inspiration from various elements of https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction,
              particularly the data_utils.py and data_loaders.py files
        """
        BINARY_NEG_LABEL = 0
        BINARY_POS_LABEL = 1

        full_signal = scio.loadmat(file_path)
        reliably_active_portion_of_full_signal = np.vstack(
            [full_signal[channel_key] for channel_key in self.channel_ids_to_keep]
        )
        labels = []
        if self.num_labels_per_sample != 1:
            raise NotImplementedError(
                "CPNEFluoxetineDataset.draw_sample: currently only supports one label assignment per data sample"
            )
        else:
            if self.task_id in ["RP", "TS"]:
                labels = [randint(BINARY_NEG_LABEL, BINARY_POS_LABEL)]
            elif self.task_id == "BehavioralFluoxetine":
                if "HomeCage" in file_path:
                    labels = [0]  # homecage label == 0
                elif "OFT" in file_path:
                    labels = [1]  # OFT label == 1
                elif "DrugRecording" in file_path:
                    # DrugRecording label in [2,3]
                    labels = [self.get_class_info_for_data_file(file_path)]
                else:
                    raise ValueError(
                        "CPNEFluoxetineDataset.draw_sample: the provided file_path=="
                        + file_path
                        + " does not have the required structure"
                    )
            else:
                raise ValueError(
                    "CPNEFluoxetineDataset.draw_sample: the provided self.task_id=="
                    + self.task_id
                    + " is not supported"
                )

        inputs = []
        input_starts = []
        for window_num in range(self.num_inputs_per_sample):
            sampled_window = None
            if self.task_id == "BehavioralFluoxetine":
                sampled_window, _ = randomly_sample_window_from_signal(
                    reliably_active_portion_of_full_signal,
                    self.task_variable_parameter_args.window_len,
                )

            elif self.task_id == "RP":
                curr_window_type = None
                curr_anchor_start = None
                curr_tpos = None
                curr_tneg = None
                if window_num == 0:
                    curr_window_type = "anchor"
                elif window_num == 1:
                    curr_window_type = "other"
                    curr_anchor_start = input_starts[0]
                    if labels[0] == BINARY_NEG_LABEL:
                        curr_tneg = self.task_variable_parameter_args.tneg
                    elif (
                        labels[0] == BINARY_POS_LABEL
                    ):  # elif labels[1] == BINARY_POS_LABEL:
                        curr_tpos = self.task_variable_parameter_args.tpos
                    else:
                        raise ValueError(
                            "CPNEFluoxetineDataset.draw_sample: unsupported label for RP task requested"
                        )
                else:
                    raise ValueError(
                        "CPNEFluoxetineDataset.draw_sample: too many windows requested for RP task"
                    )

                sampled_window, start_ind = sample_window_from_signal_for_RP_task(
                    reliably_active_portion_of_full_signal,
                    self.task_variable_parameter_args.window_len,
                    window_type=curr_window_type,
                    anchor_start=curr_anchor_start,
                    tpos=curr_tpos,
                    tneg=curr_tneg,
                )
                input_starts.append(start_ind)

            else:
                raise NotImplementedError(
                    "CPNEFluoxetineDataset.draw_sample: window sampling not implemented for self.task_id == "
                    + self.task_id
                )

            filtered_window = (
                mne.filter.filter_data(  # apply a 4th order butterworth filter
                    data=sampled_window,
                    sfreq=self.task_variable_parameter_args.fs,
                    l_freq=self.task_variable_parameter_args.l_freq,
                    h_freq=self.task_variable_parameter_args.h_freq,
                    method=self.task_variable_parameter_args.filter_method,
                    fir_window=self.task_variable_parameter_args.fir_window,
                )
            )
            inputs.append(
                filtered_window[:, 0::1]
            )  # remove the downsample (fs=1000 -> 250Hz downsample) and record as input

        sample = tuple(inputs + labels)
        return sample

    def cache_samples_for_current_split_fold(
        self,
        fold_individs,
        all_data_file_paths,
        num_samps_across_files,
        fold_save_dir,
        max_num_samps_per_subset_file,
    ):
        """
        CPNEFluoxetineDataset.get_number_of_samples_to_draw_from_each_file: determines how many samples to assign to draw from each file
         - Inputs:
            * fold_individs (list): a list of individual ids that can be used to filter source data files for inclusion into the current split fold
            * all_data_file_paths (list): a list corresonding to all available data files in the source dataset
            * num_train_samps_across_files (int): the total number of samples to be drawn from all files combined
            * fold_save_dir (str): the directory to which all split fold subset files should be saved
            * max_num_samps_per_subset_file (int): the maximum number of samples to be included in each split fold subset file
         - Outputs:
            * source_file_paths (list): a list of file paths used to populate the current split fold
            * num_samps_per_file (list): a list (of int values) representing how many samples were drawn from each file, with each int corresponding to a path in source_file_paths
         - Usage:
            * CPNEFluoxetineDataset.preprocess_and_cache_data: uses CPNEFluoxetineDataset.get_number_of_samples_to_draw_from_each_file prior to drawing samples
        """
        source_file_paths = [
            x for x in all_data_file_paths for y in fold_individs if y in x
        ]
        shuffle(source_file_paths)

        num_samps_per_source_file = [
            len(source_file_paths) // num_samps_across_files
            for _ in range(len(source_file_paths))
        ]
        for i in range(num_samps_across_files % len(source_file_paths)):
            num_samps_per_source_file[i] += 1

        curr_subset_id_counter = 0
        curr_subset = []
        curr_subset_save_path = os.sep.join(
            [fold_save_dir, "subset" + curr_subset_id_counter + ".pkl"]
        )
        for file_path, num_samps_needed in zip(
            source_file_paths, num_samps_per_source_file
        ):
            for i in range(num_samps_needed):
                # draw sample
                curr_subset.append(self.draw_sample(file_path))

                # check if current subset needs to be cached
                if len(curr_subset) == max_num_samps_per_subset_file:
                    with open(curr_subset_save_path, "wb") as outfile:
                        pkl.dump(curr_subset, outfile)
                    # initialize new subset
                    curr_subset_id_counter += 1
                    curr_subset = []
                    curr_subset_save_path = os.sep.join(
                        [fold_save_dir, "subset" + curr_subset_id_counter + ".pkl"]
                    )

        return source_file_paths, num_samps_per_source_file

    def preprocess_and_cache_data(self, data_source_specific_args):
        """
        CPNEFluoxetineDataset.load_cached_preprocessed_dataset: shuffles, splits, and caches CPNE Fluoxetine data from original source directory
         - Inputs:
            * data_source_specific_args (argparse obj): arguments for preprocessing the original CPNE Fluoxetine dataset, including
               - args.original_data_source_dir (str): the directory containing the original CPNE Fluoxetine dataset
         - Outputs:
            * */cached_samples*.pkl (cached list): pickle files containing lists of cached samples, with each sample formatted as (x1, x2, ..., xn, label_y)
            * cached_data_stats_and_params.pkl (cached dict): pickle file containing dict of info related to how the cached data was formatted/built
         - Usage:
            * PreprocessedDataset.__init__: uses CPNEFluoxetineDataset.preprocess_and_cache_data when self.data_save_directory is empty
        """
        # access source data set and identify individuals
        individual_ids = set()
        all_available_chans = set()
        potentially_inactive_chans = set()
        max_num_chans_in_single_recording = None
        min_num_chans_in_single_recording = None
        all_data_file_paths = []

        for environ_dir in os.listdir(
            data_source_specific_args.original_data_source_dir
        ):
            curr_environ_path = os.sep.join(
                [data_source_specific_args.original_data_source_dir, environ_dir]
            )
            # track individual ids
            for data_file_name in os.listdir(
                os.sep.join([curr_environ_path, "Data"])
            ):  # folder structure specific to CPNEFluoxetine
                individual_ids.add(data_file_name.split("_")[0])
                all_data_file_paths.append(
                    os.sep.join([curr_environ_path, "Data", data_file_name])
                )

            # track channel ids according to their active/inactive status - this will likely be a CPNE-specific portion of code
            if environ_dir in ["DrugRecording", "OFT"]:
                for chan_file_name in os.listdir(
                    os.sep.join([curr_environ_path, "CHANS"])
                ):  # folder structure specific to CPNEFluoxetine
                    curr_chan_file_path = os.sep.join(
                        [curr_environ_path, "CHANS", chan_file_name]
                    )
                    curr_chan_specs = scio.loadmat(
                        curr_chan_file_path
                    )  # see https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction/blob/main/data_utils.py
                    num_chans_in_curr_recording = curr_chan_specs["CHANACTIVE"].shape[0]

                    # track max and min number of channels per recording to determine if they have unified format
                    if (
                        max_num_chans_in_single_recording is None
                        or num_chans_in_curr_recording
                        > max_num_chans_in_single_recording
                    ):
                        max_num_chans_in_single_recording = num_chans_in_curr_recording
                    if (
                        min_num_chans_in_single_recording is None
                        or num_chans_in_curr_recording
                        > min_num_chans_in_single_recording
                    ):
                        min_num_chans_in_single_recording = num_chans_in_curr_recording

                    for i in range(num_chans_in_curr_recording):
                        if curr_chan_specs["CHANACTIVE"][i, 0] == 1:
                            all_available_chans.add(
                                curr_chan_specs["CHANNAMES"][0, 0, 0]
                            )
                        else:
                            potentially_inactive_chans.add(
                                curr_chan_specs["CHANNAMES"][0, 0, 0]
                            )

        individual_ids = list(individual_ids)
        assert (
            max_num_chans_in_single_recording == min_num_chans_in_single_recording
        )  # sanity check - if this requirement isn't satisfied, re-write the above loop or reformat source data
        self.channel_ids_to_keep = [
            x
            for x in list(all_available_chans)
            if x not in list(potentially_inactive_chans)
        ]

        self.data_splits_by_individual_id = self.split_individual_ids_in_dataset(
            individual_ids
        )

        # create index map for each sample in the dataset (without storing the samples yet) - indices will likely need to map to a file / location for later access
        [
            self.num_total_train_samples_per_split,
            self.num_total_val_samples_per_split,
            self.num_total_test_samples,
        ] = data_source_specific_args.nums_of_samples_per_split_by_fold
        # self.window_size = data_source_specific_args.window_size
        self.max_num_samps_per_subset_file = (
            data_source_specific_args.max_num_samps_per_subset_file
        )

        holdout_set_has_been_cached = False
        split_subset_info = {}
        curr_test_source_file_paths = None
        curr_test_num_samps_per_source_file = None
        for curr_split_num, [
            train_split_individs,
            val_split_individs,
            test_split_individs,
        ] in enumerate(self.data_splits_by_individual_id):
            # TO-DO:
            # call self.cache_samples_for_current_split_fold for each of train/val/test and store returned values
            (
                curr_train_source_file_paths,
                curr_train_num_samps_per_source_file,
            ) = self.cache_samples_for_current_split_fold(
                train_split_individs,
                all_data_file_paths,
                self.num_total_train_samples_per_split,
                self.data_save_directory + os.sep + "train" + curr_split_num,
                self.max_num_samps_per_subset_file,
            )
            (
                curr_val_source_file_paths,
                curr_val_num_samps_per_source_file,
            ) = self.cache_samples_for_current_split_fold(
                val_split_individs,
                all_data_file_paths,
                self.num_total_val_samples_per_split,
                self.data_save_directory + os.sep + "validation" + curr_split_num,
                self.max_num_samps_per_subset_file,
            )
            if not holdout_set_has_been_cached:
                (
                    curr_test_source_file_paths,
                    curr_test_num_samps_per_source_file,
                ) = self.cache_samples_for_current_split_fold(
                    test_split_individs,
                    all_data_file_paths,
                    self.num_total_test_samples,
                    self.data_save_directory + os.sep + "test" + curr_split_num,
                    self.max_num_samps_per_subset_file,
                )
                holdout_set_has_been_cached = True

            split_subset_info["split" + str(curr_split_num)] = {
                "train": {
                    "train_source_file_paths": curr_train_source_file_paths,
                    "train_num_samps_per_source_file": curr_train_num_samps_per_source_file,
                },
                "validation": {
                    "val_source_file_paths": curr_val_source_file_paths,
                    "val_num_samps_per_source_file": curr_val_num_samps_per_source_file,
                },
                "test": {
                    "test_source_file_paths": curr_test_source_file_paths,
                    "test_num_samps_per_source_file": curr_test_num_samps_per_source_file,
                },
            }
            pass

        with open(
            self.data_save_directory + os.sep + "cached_data_stats_and_params.pkl", "wb"
        ) as outfile:
            pkl.dump(
                {
                    "all_data_file_paths": all_data_file_paths,
                    "individual_ids": individual_ids,
                    "channel_ids_to_keep": self.channel_ids_to_keep,
                    "data_splits_by_individual_id": self.data_splits_by_individual_id,
                    "data_source_specific_args": data_source_specific_args,
                    "split_subset_info": split_subset_info,
                    "task_variable_parameter_args": self.task_variable_parameter_args,
                },
                outfile,
            )
        pass


if __name__ == "__main__":
    """
    preprocess_CPNEFluoxetine_data.__main__: shuffles, splits, and caches CPNE Fluoxetine data based on provided arguments
    - Inputs:
       * cached_args_file (argparse obj pickle file): arguments for preprocessing the original CPNE Fluoxetine dataset, including
          - cached_args.root_save_directory (str): the directory containing the original CPNE Fluoxetine dataset
          - cached_args.data_split_type (str): one of ['standardSplit', 'crossValSplit'] denoting the type of dataset being curated
          - cached_args.task_id (str): one of the id's present in self.get_hyperparams_for_task cases denoting the types of tasks being modeled in the current dataset
          - cached_args.data_source_id (str): an identifier of the original dataset (e.g., 'TUAB' or 'CPNEFluoxetine')
          - cached_args.date_of_data_split (str): a YYYYMMDD-formated string denoting the date of dataset curation/preprocessing
          - cached_args.task_variable_parameter_args (argparse obj): taski-specific hyperparameters (e.g., tau-pos for RP task) which may need tuning, default=None,
          - cached_args.data_source_specific_args (argparse obj): arguments for preprocessing the CPNE Fluoxetine dataset, default=None,
          - cached_args.num_splits (int): the number of train-val-test splits to make (esp. for cross-validation), default=1
          - cached_args.holdout_set (boolean): whether a test split needs to be stored as a holdout set, default=False
          - cached_args.split_ratios (tuple): a tuple containing the ratios (float) of each split data subset to be assigned to the train, validation, and test sets, respectively, default=(0.7, 0.2, 0.1)
    - Outputs:
       * */cached_samples*.pkl (cached list): pickle files containing lists of cached samples, with each sample formatted as (x1, x2, ..., xn, label_y)
    - Usage:
       * PreprocessedDataset.__init__: uses CPNEFluoxetineDataset.preprocess_and_cache_data when self.data_save_directory is empty
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cached_args_file",
        type=str,
        default="cached_args_preprocess_CPNEFluoxetine_data.pkl",
        help="/path/to/cached_args.pkl should contain an argparse.ArgumentParser() with arguments structured as in the __main__ doc string",
    )
    parser.add_argument(
        "-n", default=None, help="IGNORE THIS ARGUMENT - IT'S A BANDAID"
    )
    args = parser.parse_args()

    print("<<< MAIN: START")

    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print("<<< KICKING OFF TASKID == ", taskID)

    with open(args.cached_args_file, "rb") as infile:
        cached_args = pkl.load(infile)
        print("<<< cached_args == ", cached_args)
        _ = CPNEFluoxetineDataset(
            cached_args.root_save_directory,
            cached_args.data_split_type,
            cached_args.task_id,
            cached_args.data_source_id,
            cached_args.date_of_data_split,
            cached_args.task_variable_parameter_args,
            cached_args.data_source_specific_args,
            cached_args.num_splits,
            cached_args.holdout_set,
            cached_args.split_ratios,
        )

    print("<<< MAIN: DONE RUNNING TASKID == ", taskID, "!!!")
    pass

