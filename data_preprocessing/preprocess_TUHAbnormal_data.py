# preprocess_TUHAbnormal_data.py: a file defining TUH Abnormal dataset-specific preprocessing classes and functions
# SEE LICENSE STATEMENT AT THE END OF THE FILE

# args2.data_source_path =  

# arg.data_source_specific_args = args
# -> cached_args["v2"] = args (do that for all 3 tasks)

#-> # with open ("/Users/akira/CodeProjects/updated_cdisn/data_preprocessing/cached_args_preprocess_TUHAbnormal_data.pkl" wb) as infile:
#pkl.dump("cached_args[v2]")

#

# 'behavioralTUAB100percent_v2': Namespace(data_source_id='TUAB', data_source_specific_args=Namespace(data_source_path='/work/xd62/edf', max_num_samps_per_subset_file=100000, nums_of_samples_per_split_by_fold=[868400, 217200, 110400]), data_split_type='crossValSplit', date_of_data_split='02182022', max_num_bad_edf_filtering_samples=None, num_edf_filtering_samples=None, num_splits=5, root_save_directory='/work/xd62/preprocessedTUABDatasets', split_ratios=(0.8, 0.2, 0.0), task_id='BehavioralTUAB', task_variable_parameter_args=Namespace(filter_method=None, fir_window=None, fs=None, h_freq=None, l_freq=None, throttle_ratio=1.0, window_len=600, tpos=None, tneg=None)),

#  'rpTUAB100percent_v2': Namespace(data_source_id='TUAB', data_source_specific_args=Namespace(data_source_path='/work/xd62/edf', max_num_samps_per_subset_file=100000, nums_of_samples_per_split_by_fold=[868400, 217200, 110400]), data_split_type='crossValSplit', date_of_data_split='02182022', max_num_bad_edf_filtering_samples=None, num_edf_filtering_samples=None, num_splits=1, root_save_directory='/work/xd62/preprocessedTUABDatasets', split_ratios=(0.7, 0.2, 0.1), task_id='RP', task_variable_parameter_args=Namespace(filter_method=None, fir_window=None, fs=None, h_freq=None, l_freq=None, throttle_ratio=1.0, window_len=600, tpos=2000, tneg=31600)),

#  'tsTUAB100percent_v2': Namespace(data_source_id='TUAB', data_source_specific_args=Namespace(data_source_path='/work/xd62/edf', max_num_samps_per_subset_file=100000, nums_of_samples_per_split_by_fold=[868400, 217200, 110400]), data_split_type='crossValSplit', date_of_data_split='02182022', max_num_bad_edf_filtering_samples=None, num_edf_filtering_samples=None, num_splits=1, root_save_directory='/work/xd62/preprocessedTUABDatasets', split_ratios=(0.7, 0.2, 0.1), task_id='TS', task_variable_parameter_args=Namespace(filter_method=None, fir_window=None, fs=None, h_freq=None, l_freq=None, throttle_ratio=1.0, window_len=600, tpos=3100, tneg=63000)),


# dependency import statements
import os
import argparse
import pickle as pkl
import mne
import numpy as np
from random import shuffle, randint
import random
from data_preprocessing_utils import (
    randomly_sample_window_from_signal,
    sample_window_from_signal_for_RP_task,
    sample_window_from_signal_for_TS_task,
)
from tuab_edf_utils import nedc_select_channels
# import pyedflib
from tuab_edf_utils import nedc_load_edf, nedc_get_pos, nedc_apply_montage
import uuid
import random
from data_preprocessing_utils import PreprocessedDataset

# get the file (row signal) sample windows from it, and store each window into train, vali, or test dataset. edf file -> numpy array
# use num_inputs_per_sample, num_lables_per_sample for sampling windows
# open EFL file and access eval and train
# for eval,
# use train dataset for train and validation, and use eval dataset for test.

# step1: group individuals regardles of RPTS or Behavioral TUAB(read the readme of the dataset) -> make two individual sets (train and test separately)

# step2: Split data: train, validation, test
# split id name (train and validation separately) and do for loop through all the data file inside the train data, and if the maximum number hits, put them into the pkl file.

# step3 Keep the same person together even if its split into normal or abnormal Balance data(same number of normal and abnormal data, drop some data after splitting into train validation) and track whethereits normal or abnormal if behaviroral
# draw sampling depending on the task and lable positive or negative for each data.
# if its behavioral, you label it either normal or abnormal. (look at the previous file)
# RP/TS ->
# Behavioral TUAB -> Include abnormal and normal same person data -> group the file together
# window is input, output is vector
# if num samples drawn  >= args.m, save data.pkl to make sure the data is not too big to use


# split is doing for dataset,
class TUHAbnormalDataset(PreprocessedDataset):
    def __init__(
        self,
        root_save_directory,
        data_split_type,
        task_id,
        data_source_id,
        date_of_data_split,
        task_variable_parameter_args=None,
        data_source_specific_args=None,
        num_splits=1,  # holdout_set=True,
        split_ratios=(0.8, 0.2, 0), # num_edf_filtering_samples=5, # max_num_bad_edf_filtering_samples=2,
    ):
        # assert max_num_bad_edf_filtering_samples <= num_edf_filtering_samples
        # self.num_edf_filtering_samples = num_edf_filtering_samples
        # self.max_num_bad_edf_filtering_samples = max_num_bad_edf_filtering_samples
        self.unique_temp_file_id = str(uuid.uuid1()) # see https://stackoverflow.com/questions/534839/how-to-create-a-guid-uuid-in-python as well as https://stackoverflow.com/questions/1785503/when-should-i-use-uuid-uuid1-vs-uuid-uuid4-in-python
        
        # set params for cleaning TUAB edf files - see page 13 of https://arxiv.org/pdf/2007.16104.pdf
        self.approx_names_of_good_channels = [
            "FP1", 
            "FP2", 
            "F7", 
            "F8", 
            "F3", 
            "FZ", 
            "F4", 
            "A1", 
            "T3", 
            "C3", 
            "CZ", 
            "C4", 
            "T4", 
            "A2", 
            "T5", 
            "P3", 
            "PZ", 
            "P4", 
            "T6", 
            "O1", 
            "O2"
        ]
        self.downsample_freq = 100 # Hz
        self.lower_crop_bound = 60*self.downsample_freq # following Banville et al., we crop first minute from all recordings
        self.upper_crop_bound = 20*self.lower_crop_bound # likewise, we limit each recording to be no longer than 20 mins
        self.upper_amplitude_clip_bound = 800 # microvolts
        self.lower_amplitude_clip_bound = -800 # microvolts
        self.sig_variation_reject_threshold = 1 # microvolt
        self.downsample_fir_window = 'hamming' # maybe boxcar? see https://mne.tools/stable/generated/mne.filter.resample.html

        # define montage for de-noising TUAB sigs - see https://par.nsf.gov/servlets/purl/10199699
        self.matchmode = 'partial'
        self.denoising_montage = [ # see nedc_parse_montage function in tuab_edf_utils.py for structure
            [0, "FP1-F7", "EEG FP1-REF", "EEG F7-REF"], # see Appendix A of https://par.nsf.gov/servlets/purl/10199699
            [1, "F7-T3", "EEG F7-REF", "EEG T3-REF"],
            [2, "T3-T5", "EEG T3-REF", "EEG T5-REF"],
            [3, "T5-O1", "EEG T5-REF", "EEG O1-REF"],
            [4, "FP2-F8", "EEG FP2-REF", "EEG F8-REF"],
            [5, "F8-T4", "EEG F8-REF", "EEG T4-REF"],
            [6, "T4-T6", "EEG T4-REF", "EEG T6-REF"],
            [7, "T6-O2", "EEG T6-REF", "EEG O2-REF"],
            [8, "A1-T3", "EEG A1-REF", "EEG T3-REF"],
            [9, "T3-C3", "EEG T3-REF", "EEG C3-REF"],
            [10, "C3-CZ", "EEG C3-REF", "EEG CZ-REF"],
            [11, "CZ-C4", "EEG CZ-REF", "EEG C4-REF"],
            [12, "C4-T4", "EEG C4-REF", "EEG T4-REF"],
            [13, "T4-A2", "EEG T4-REF", "EEG A2-REF"],
            [14, "FP1-F3", "EEG FP1-REF", "EEG F3-REF"],
            [15, "F3-C3", "EEG F3-REF", "EEG C3-REF"],
            [16, "C3-P3", "EEG C3-REF", "EEG P3-REF"],
            [17, "P3-O1", "EEG P3-REF", "EEG O1-REF"],
            [18, "FP2-F4", "EEG FP2-REF", "EEG F4-REF"],
            [19, "F4-C4", "EEG F4-REF", "EEG C4-REF"],
            [20, "C4-P4", "EEG C4-REF", "EEG P4-REF"],
            [21, "P4-O2", "EEG P4-REF", "EEG O2-REF"],
        ]

        # super().__init__() # see https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        # super(PreprocessedDataset, self).__init__(
        super().__init__(
            root_save_directory,
            data_split_type,
            task_id,
            data_source_id,
            date_of_data_split,
            task_variable_parameter_args,
            data_source_specific_args,
            num_splits,  # holdout_set,
            split_ratios,
        )
        pass

    def get_edf_filenames_to_filter(
        self, data_source_path#: str
    ):  # data_source_specific_args: str):
        """
        Outputs: A list of edf file from all the directories (string)
        """

        temp_path_list = []
        edf_file_list = []

        for directory in os.listdir(data_source_path):  # data_source_specific_args):
            # remove irrelevant Mac file
            if not directory.startswith("."):
                curr_environ_path: str = os.sep.join(
                    [
                        data_source_path,
                        directory,
                    ]  # data_source_specific_args, directory]
                )
                curr_environ_path_normal: str = os.sep.join(
                    [curr_environ_path, "normal"]
                )
                curr_environ_path_abnormal: str = os.sep.join(
                    [curr_environ_path, "abnormal"]
                )
                # make abnormal and normal path list for both train and eval
                path_list = [
                    curr_environ_path_normal + "/01_tcp_ar",
                    curr_environ_path_abnormal + "/01_tcp_ar",
                ]
                # temp_path_list.append(path_list[0])
                # temp_path_list.append(path_list[1])
                temp_path_list = temp_path_list + path_list

        complete_path_list = []
        for path in temp_path_list:
            directories = os.listdir(path)
            for directory in directories:
                if "." not in directory:
                    curr_path = os.sep.join([path, directory])
                    deeper_directories = os.listdir(curr_path)
                    for directory in deeper_directories:
                        complete_path = os.sep.join([curr_path, directory])

                        complete_path_list.append(complete_path)

        temp_path_list = []
        for path in complete_path_list:
            if "." not in path:
                directories = os.listdir(path)
                for directory in directories:
                    new_path = os.sep.join([path, directory])
                    temp_path_list.append(new_path)

        for path in temp_path_list:
            if "." not in path:
                print("path == ", path)
                edf_files = [x for x in os.listdir(path) if ".edf" in x]
                # print("edf_files == ", edf_files)
                assert len(edf_files) == 1
                # edf_file_list.append(edf_files[0])
                edf_file_list.append(os.sep.join([path, edf_files[0]]))

        return edf_file_list

    ##################################################################################################################################
    # TUAB EDF FILE FILTERING CODE - THE FIRST IMPLEMENTATION IS COMMENTED OUT FOR BACKUP, BUT THE USEFUL IMPLEMENTATION COMES AFTER

    # def filter_data(self, data_source_specific_args):

    #     # get all the edf files
    #     edf_file_list = self.get_edf_filenames_to_filter(
    #         data_source_specific_args.data_source_path
    #     )
    #     good_edf_files = []
    #     good_channels = []
    #     # channels_in_configuration = None
    #     total_num_edfs = 0
    #     # num_eval_edfs_accepted = 0 # FOR DEBUGGING PURPOSES
    #     for j, edf_file in enumerate(edf_file_list):
    #         # if num_eval_edfs_accepted > 10 and 'eval' in edf_file: # FOR DEBUGGING PURPOSES
    #         #     pass
    #         # else:
    #         ################################################################
    #         # open edf file - see https://pyedflib.readthedocs.io/en/latest/
    #         f = pyedflib.EdfReader(edf_file)
    #         n = f.signals_in_file
    #         if j == 0:
    #             good_channels = [i for i in range(n)]
    #             # channels_in_configuration = n
    #         max_sig_len_in_file = np.max([f.readSignal(i).shape[0] for i in range(n)])
    #         # signal_labels = f.getSignalLabels()
    #         # orig_signal = np.zeros((n, f.getNSamples()[0]))
    #         orig_signal = np.zeros((n, max_sig_len_in_file))
    #         curr_good_channels = []
    #         for i in np.arange(n):
    #             # print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.filter_data: READING IN SIGNAL WITH SHAPE == ", f.readSignal(i).shape)
    #             # print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.filter_data: ATTEMPTING TO FIT INTO ARRAY WITH SHAPE == ", orig_signal[i,:].shape)
    #             if f.readSignal(i).shape[0] == max_sig_len_in_file:
    #                 orig_signal[i, :] = f.readSignal(i)
    #                 curr_good_channels.append(i)
    #         ################################################################

    #         bad_recording_counter = 0
    #         for i in range(self.num_edf_filtering_samples):
    #             sampled_window, _ = randomly_sample_window_from_signal(
    #                 orig_signal,
    #                 self.task_variable_parameter_args.window_len,
    #                 i, # random seed to ensure different datasets filter based on same samples
    #             )
    #             if np.sum(sampled_window) == 0:
    #                 bad_recording_counter += 1

    #         if (
    #             bad_recording_counter < self.max_num_bad_edf_filtering_samples
    #             # and len(curr_good_channels) > n // 2
    #         ):  # edf recording is good if there aren't too many 'bad recordings' and more than half of the recordings lasted for the full session
    #             good_channels = [x for x in good_channels if x in curr_good_channels]
    #             pickled_file_name = (
    #                 edf_file[:-4] + self.unique_temp_file_id + ".pkl"
    #             )  # save pickled version of signal for later use (i.e., __getitem__ calls)
    #             with open(pickled_file_name, "wb") as outfile:
    #                 pkl.dump(orig_signal, outfile)
    #             good_edf_files.append(edf_file)

    #             # if num_eval_edfs_accepted <= 10 and 'eval' in edf_file: # FOR DEBUGGING PURPOSES
    #             #     num_eval_edfs_accepted += 1

    #         total_num_edfs += 1

    #         if len(good_channels) < 3:
    #             raise ValueError("Not enough channels are being preserved during TUAB signal filtering - consider lowering number of acceptable channels or making filtering less strict.")
            
    #         # # FOR DEBUGGING PURPOSES
    #         # if len(good_edf_files) > 20 and len(good_channels) >= 10:
    #         #     print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.filter_data: BREAKING FOR DEBUGGING PURPOSES")
    #         #     break

    #     assert (
    #         len(good_edf_files) > 20
    #     )  # 20 is an arbitrarily set minimum value as to how many clean edfs we should find
    #     assert (
    #         len(good_channels) >= 10
    #     )  # 10 is an arbitrarily set minimum value as to how many channels should be robust/uninterrupted across the dataset
    #     self.good_edf_files = good_edf_files
    #     self.good_channels = good_channels
    #     print(
    #         "TUHAbnormalDataset.filter_data: ",
    #         len(good_edf_files),
    #         " of ",
    #         total_num_edfs,
    #         " edf files were deemed relevant enough to use in experiments.",
    #     )
    #     pass

    def filter_data(self, data_source_specific_args):

        # get all the edf files
        edf_file_list = self.get_edf_filenames_to_filter(
            data_source_specific_args.data_source_path
        )

        good_edf_files = []
        # channels_in_configuration = None
        total_num_edfs = 0
        # num_eval_edfs_accepted = 0 # FOR DEBUGGING PURPOSES
        for j, edf_file in enumerate(edf_file_list):
            # if num_eval_edfs_accepted > 10 and 'eval' in edf_file: # FOR DEBUGGING PURPOSES
            #     pass
            # else:            
            # load original signal
            orig_samp_freqs, orig_sig, orig_labels = nedc_load_edf(edf_file)

            # grab only the channels named in self.approx_names_of_good_channels
            chans_to_keep_indices = []
            for chan_name in self.approx_names_of_good_channels:
                ind = nedc_get_pos(chan_name, orig_labels, self.matchmode)
                if ind < 0:
                    print("TUHAbnormalDataset.filter_data: IMPORTANT CHANNEL MISSING FROM EDF")
                    print("TUHAbnormalDataset.filter_data: chan_name == ", chan_name)
                    print("TUHAbnormalDataset.filter_data: orig_labels == ", orig_labels)
                    print("TUHAbnormalDataset.filter_data: edf_file == ", edf_file)
                    raise ValueError("TUHAbnormalDataset.filter_data: IMPORTANT CHANNEL MISSING FROM EDF, leaving ind < 0")
                else:
                    chans_to_keep_indices.append(ind)

            assert len(chans_to_keep_indices) == len(self.approx_names_of_good_channels)
            orig_samp_freqs = [orig_samp_freqs[ind] for ind in chans_to_keep_indices] # orig_samp_freqs[chans_to_keep_indices]
            orig_sig = [orig_sig[ind] for ind in chans_to_keep_indices] # orig_sig[chans_to_keep_indices]
            orig_labels = [orig_labels[ind] for ind in chans_to_keep_indices] # orig_labels[chans_to_keep_indices]

            # apply montage to denoise signals
            montage_samp_freqs, montage_sig, montage_labels = nedc_apply_montage(None, orig_samp_freqs, orig_sig, orig_labels, self.denoising_montage, self.matchmode)

            # downsample signals
            curr_num_chans = len(montage_sig)
            for i in range(curr_num_chans):
                curr_samp_freq = montage_samp_freqs[i]
                curr_decimation_factor = curr_samp_freq / self.downsample_freq
                if curr_decimation_factor > 1.0:
                    montage_sig[i] = mne.filter.resample(montage_sig[i], up=1.0, down=curr_decimation_factor, npad="auto", window=self.downsample_fir_window)
                else:
                    montage_sig[i] = mne.filter.resample(montage_sig[i], up=1./curr_decimation_factor, down=1.0, npad="auto", window=self.downsample_fir_window)

            # crop signals
            min_chan_record_len_in_file = np.min([montage_sig[i].shape[0] for i in range(curr_num_chans)])
            curr_crop_upper_bound = min(min_chan_record_len_in_file, self.upper_crop_bound)
            new_sig_window_len = curr_crop_upper_bound - self.lower_crop_bound
            cropped_sig = np.zeros((curr_num_chans, new_sig_window_len))
            for i in range(curr_num_chans):
                cropped_sig[i,:] = montage_sig[i][self.lower_crop_bound:curr_crop_upper_bound]

            # clip signals
            pos_mask = cropped_sig >= self.upper_amplitude_clip_bound
            neg_mask = cropped_sig <= self.lower_amplitude_clip_bound
            pos_diffs = self.upper_amplitude_clip_bound - cropped_sig
            neg_diffs = self.lower_amplitude_clip_bound - cropped_sig
            clean_sig = cropped_sig + pos_mask*pos_diffs + neg_mask*neg_diffs

            # reject bad signals
            reject_signal = False
            for i in range(curr_num_chans):
                if np.max(clean_sig[i]) - np.min(clean_sig[i]) < self.sig_variation_reject_threshold:
                    reject_signal = True

            # save signals
            if not reject_signal:
                pickled_file_name = (edf_file[:-4] + self.unique_temp_file_id + ".npy")  # save pickled version of signal for later use (i.e., __getitem__ calls)
                with open(pickled_file_name, "wb") as outfile:
                    pkl.dump(clean_sig, outfile)
                good_edf_files.append(edf_file)
            
                # if num_eval_edfs_accepted <= 10 and 'eval' in edf_file: # FOR DEBUGGING PURPOSES
                #     num_eval_edfs_accepted += 1

                total_num_edfs += 1

            # # FOR DEBUGGING PURPOSES
            # if len(good_edf_files) > 20: 
            #     print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.filter_data: BREAKING FOR DEBUGGING PURPOSES")
            #     break

        assert (
            len(good_edf_files) > 20
        )  # 20 is an arbitrarily set minimum value as to how many clean edfs we should find

        self.good_edf_files = good_edf_files
        print(
            "TUHAbnormalDataset.filter_data: ",
            len(good_edf_files),
            " of ",
            total_num_edfs,
            " edf files were deemed relevant enough to use in experiments.",
        )
        pass
    ##################################################################################################################################

    def get_individuals(self, data_source_path: str):  # specific_args: str):

        """
        - Inputs:
            * data_source_specific_args (argparse obj): arguments for preprocessing the original CPNE Fluoxetine dataset, including
                - args.original_data_source_dir (str): the directory containing the original CPNE Fluoxetine dataset

        - Outputs: list of individual ids (string)
        """
        individual_ids_train = set()
        individual_ids_test = set()
        all_data_file_paths = []

        for directory in os.listdir(data_source_path):  # data_source_specific_args):
            # remove irrelevant Mac file
            if not directory.startswith("."):
                # print("get_individuals: ITERATING OVER directory == ", directory)
                curr_environ_path: str = os.sep.join(
                    [
                        data_source_path,
                        directory,
                    ]  # data_source_specific_args, directory]
                )
                curr_environ_path_normal: str = os.sep.join(
                    [curr_environ_path, "normal"]
                )
                curr_environ_path_abnormal: str = os.sep.join(
                    [curr_environ_path, "abnormal"]
                )
                # make abnormal and normal path list for both train and eval
                path_list = [
                    curr_environ_path_normal + "/01_tcp_ar",
                    curr_environ_path_abnormal + "/01_tcp_ar",
                ]

                for idx, path in enumerate(path_list):
                    # three digits directory list per each normal/abnormal/eval/train directory
                    threedigits_list = os.listdir(path)
                    for threedigits in threedigits_list:
                        if not threedigits.startswith("."):
                            subdirectories = os.listdir(
                                os.sep.join([path, threedigits])
                            )
                            for individual_id in subdirectories:
                                # track individual ids for test and train separately
                                curr_path = os.sep.join(
                                    [path, threedigits, individual_id]
                                )
                                is_good_path = False
                                for good_edf_file_path in self.good_edf_files:
                                    if curr_path in good_edf_file_path:
                                        is_good_path = True
                                        break
                                if is_good_path:
                                    if "eval" in path.split("/"):
                                        individual_ids_test.add(individual_id)
                                        all_data_file_paths.append(curr_path)
                                    else:
                                        individual_ids_train.add(individual_id)
                                        all_data_file_paths.append(curr_path)

        individual_ids_train_list = list(individual_ids_train)
        individual_ids_test_list = list(individual_ids_test)
        # print(all_data_file_paths)

        return (
            individual_ids_train_list,
            individual_ids_test_list,
            all_data_file_paths,
        )

    # def test_path(self, test_individual_ids, all_data_file_paths):
    #     test_source_file_paths = [
    #         x for x in all_data_file_paths for y in test_individual_ids if y in x
    #     ]
    #     return test_source_file_paths

    # def train_split_path(self, individual_ids_train, all_data_file_paths):
    #     # split train_data, eval_data into trainfiles and valfiles
    #     self.data_splits_by_individual_id_train = self.split_individual_ids_in_dataset(
    #         individual_ids_train
    #     )
    #     # split into train, val, 1453, 622, 1 out of 2076 in total

    #     # split train data
    #     for idx, [train_split_id, val_split_id, test_split_id] in enumerate(
    #         self.data_splits_by_individual_id_train
    #     ):
    #         train_source_file_paths = [
    #             x for x in all_data_file_paths for y in train_split_id if y in x
    #         ]
    #         shuffle(train_source_file_paths)

    #         val_source_file_paths = [
    #             x for x in all_data_file_paths for y in val_split_id if y in x
    #         ]
    #         shuffle(val_source_file_paths)
    #         test_source_file_paths = None

    #     return (train_source_file_paths, val_source_file_paths)
    def draw_sample(self, file_path):
        """
        TUHAbnormalDataset.draw_sample: draws a random sample from a data file corresponding the self.task_id
         - Inputs:
            * file_path (str): the path to the source data file from which a sample is to be drawn
         - Outputs:
            * sample (tuple): a tuple formatted as (x1, x2, ..., xn, label_y), representing a single data point for training on self.task_id, with xi shaped as (window_len, num_channels)
         - Usage:
            * TUHAbnormalDataset.cache_samples_for_current_split_fold: uses TUHAbnormalDataset.draw_sample prior to draw a single sample

        Note: this function draws inspiration from various elements of https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction,
              particularly the data_utils.py and data_loaders.py files
        """
        BINARY_NEG_LABEL = 0
        BINARY_POS_LABEL = 1

        full_signal = None
        with open(file_path, "rb") as infile:
            full_signal = np.load(infile, allow_pickle=True)
            # full_signal = full_signal[
            #     self.good_channels, :
            # ]  # index only into the channels that reliably uninterrupted across the entire dataset, as deteremined by self.filter_data()
        
        # initialize labels for current sample
        labels = []
        #TODO: new task id (behavioral label and individual ID) 
        if self.num_labels_per_sample != 1:
            if self.task_id in ["AnchoredBTUABRPTS", "NonAnchoredBTUABRPTS"]:
                assert self.num_labels_per_sample == 3
                labels = [
                    None, 
                    randint(BINARY_NEG_LABEL, BINARY_POS_LABEL), # RP label
                    randint(BINARY_NEG_LABEL, BINARY_POS_LABEL)  # TS label
                ]
                if "abnormal" in file_path:
                    labels[0] = 1  # behavioral abnormal label == 1
                elif "normal" in file_path:
                    labels[0] = 0  # behavioral normal label == 0
                else:
                    raise ValueError(
                        "TUHAbnormalDataset.draw_sample: the provided file_path=="
                        + file_path
                        + " does not have the required structure"
                    )
            else:
                raise ValueError(
                    "TUHAbnormalDataset.draw_sample: the provided self.task_id=="
                    + self.task_id
                    + " is not supported"
                )
        else:
            if self.task_id in ["RP", "TS"]:
                labels = [randint(BINARY_NEG_LABEL, BINARY_POS_LABEL)]
            #TODO: which individual that file belongs to, map individual ID to just an integer, one hot encoding
            elif self.task_id == "BehavioralTUAB":
                if "abnormal" in file_path:
                    labels = [1]  # abnormal label == 1
                elif "normal" in file_path:
                    labels = [0]  # normal label == 0
                else:
                    raise ValueError(
                        "TUHAbnormalDataset.draw_sample: the provided file_path=="
                        + file_path
                        + " does not have the required structure"
                    )
            else:
                raise ValueError(
                    "TUHAbnormalDataset.draw_sample: the provided self.task_id=="
                    + self.task_id
                    + " is not supported"
                )

        inputs = []
        input_starts = []
        #TODO: handle new task ID variable
        for window_num in range(self.num_inputs_per_sample):
            sampled_window = None
            if self.task_id == "BehavioralTUAB":
                sampled_window, _ = randomly_sample_window_from_signal(
                    full_signal,
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
                            "TUHAbnormalDataset.draw_sample: unsupported label for RP task requested"
                        )
                else:
                    raise ValueError(
                        "TUHAbnormalDataset.draw_sample: too many windows requested for RP task"
                    )

                sampled_window, start_ind = sample_window_from_signal_for_RP_task(
                    full_signal,
                    self.task_variable_parameter_args.window_len,
                    window_type=curr_window_type,
                    anchor_start=curr_anchor_start,
                    tpos=curr_tpos,
                    tneg=curr_tneg,
                )
                input_starts.append(start_ind)

            elif self.task_id == "TS":
                curr_window_type = None
                curr_anchor1_start = None
                curr_anchor2_start = None
                curr_tpos = None
                curr_tneg = None
                if window_num == 0:
                    curr_window_type = "anchor1"
                elif window_num == 1:
                    curr_window_type = "anchor2"
                    curr_anchor1_start = input_starts[0]
                    curr_tpos = self.task_variable_parameter_args.tpos
                elif window_num == 2:
                    curr_window_type = "other"
                    curr_anchor1_start = input_starts[0]
                    curr_anchor2_start = input_starts[1]
                    if labels[0] == BINARY_NEG_LABEL:
                        curr_tneg = self.task_variable_parameter_args.tneg
                    elif (
                        labels[0] == BINARY_POS_LABEL
                    ):  # elif labels[1] == BINARY_POS_LABEL:
                        curr_tpos = self.task_variable_parameter_args.tpos
                    else:
                        raise ValueError(
                            "TUHAbnormalDataset.draw_sample: unsupported label for TS task requested"
                        )
                else:
                    raise ValueError(
                        "TUHAbnormalDataset.draw_sample: too many windows requested for TS task"
                    )

                sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                    full_signal,
                    self.task_variable_parameter_args.window_len,
                    window_type=curr_window_type,
                    anchor1_start=curr_anchor1_start,
                    anchor2_start=curr_anchor2_start,
                    tpos=curr_tpos,
                    tneg=curr_tneg,
                )
                input_starts.append(start_ind)
            
            elif self.task_id == "AnchoredBTUABRPTS":
                if window_num == 0: # draw anchor
                    curr_window_type = "anchor1"
                    curr_anchor1_start = None
                    curr_anchor2_start = None
                    curr_tpos = None
                    curr_tneg = None
                    sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor1_start=curr_anchor1_start,
                        anchor2_start=curr_anchor2_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 1: # draw rp other
                    curr_tpos = None
                    curr_tneg = None
                    curr_window_type = "other"
                    curr_anchor_start = input_starts[0]
                    if labels[1] == BINARY_NEG_LABEL:
                        curr_tneg = self.task_variable_parameter_args.rp_tneg
                    elif (
                        labels[1] == BINARY_POS_LABEL
                    ):  # elif labels[1] == BINARY_POS_LABEL:
                        curr_tpos = self.task_variable_parameter_args.rp_tpos
                    else:
                        raise ValueError(
                            "TUHAbnormalDataset.draw_sample: unsupported label for RP task requested"
                        )

                    # print("AnchoredBTUABRPTS Sampling: now sampling for window_num == ", window_num)
                    # print("AnchoredBTUABRPTS Sampling: curr_tpos == ", curr_tpos)
                    # print("AnchoredBTUABRPTS Sampling: curr_tneg == ", curr_tneg)
                    # print("AnchoredBTUABRPTS Sampling: curr_window_type == ", curr_window_type)
                    # print("AnchoredBTUABRPTS Sampling: curr_anchor_start == ", curr_anchor_start)
                    # print("AnchoredBTUABRPTS Sampling: full_signal.shape == ", full_signal.shape)
                    # print("AnchoredBTUABRPTS Sampling: self.task_variable_parameter_args.window_len == ", self.task_variable_parameter_args.window_len)
                    sampled_window, start_ind = sample_window_from_signal_for_RP_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor_start=curr_anchor_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 2: # draw ts anchor2
                    curr_window_type = "anchor2"
                    curr_anchor1_start = input_starts[0]
                    curr_anchor2_start = None
                    curr_tpos = self.task_variable_parameter_args.ts_tpos
                    curr_tneg = None
                    sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor1_start=curr_anchor1_start,
                        anchor2_start=curr_anchor2_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 3: # draw ts other
                    curr_window_type = "other"
                    curr_anchor1_start = input_starts[0]
                    curr_anchor2_start = input_starts[2]
                    curr_tpos = None
                    curr_tneg = None
                    if labels[2] == BINARY_NEG_LABEL:
                        curr_tneg = self.task_variable_parameter_args.ts_tneg
                    elif (
                        labels[2] == BINARY_POS_LABEL
                    ):  # elif labels[1] == BINARY_POS_LABEL:
                        curr_tpos = self.task_variable_parameter_args.ts_tpos
                    else:
                        raise ValueError(
                            "TUHAbnormalDataset.draw_sample: unsupported label for TS task requested"
                        )

                    sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor1_start=curr_anchor1_start,
                        anchor2_start=curr_anchor2_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                else:
                    raise ValueError(
                        "TUHAbnormalDataset.draw_sample: too many windows requested for AnchoredBTUABRPTS task"
                    )

            elif self.task_id == "NonAnchoredBTUABRPTS":
                if window_num == 0: # draw random sample window
                    sampled_window, start_ind = randomly_sample_window_from_signal(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                    )
                    input_starts.append(start_ind)
                elif window_num == 1: # draw rp anchor
                    curr_tpos = None
                    curr_tneg = None
                    curr_window_type = "anchor"
                    curr_anchor_start = None
                    sampled_window, start_ind = sample_window_from_signal_for_RP_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor_start=curr_anchor_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 2: # draw rp other
                    curr_tpos = None
                    curr_tneg = None
                    curr_window_type = "other"
                    curr_anchor_start = input_starts[1]
                    if labels[1] == BINARY_NEG_LABEL:
                        curr_tneg = self.task_variable_parameter_args.rp_tneg
                    elif (
                        labels[1] == BINARY_POS_LABEL
                    ):  # elif labels[1] == BINARY_POS_LABEL:
                        curr_tpos = self.task_variable_parameter_args.rp_tpos
                    else:
                        raise ValueError(
                            "TUHAbnormalDataset.draw_sample: unsupported label for RP task requested"
                        )

                    # print("NonAnchoredBTUABRPTS Sampling: now sampling for window_num == ", window_num)
                    # print("NonAnchoredBTUABRPTS Sampling: curr_tpos == ", curr_tpos)
                    # print("NonAnchoredBTUABRPTS Sampling: curr_tneg == ", curr_tneg)
                    # print("NonAnchoredBTUABRPTS Sampling: curr_window_type == ", curr_window_type)
                    # print("NonAnchoredBTUABRPTS Sampling: curr_anchor_start == ", curr_anchor_start)
                    # print("NonAnchoredBTUABRPTS Sampling: full_signal.shape == ", full_signal.shape)
                    # print("NonAnchoredBTUABRPTS Sampling: self.task_variable_parameter_args.window_len == ", self.task_variable_parameter_args.window_len)
                    sampled_window, start_ind = sample_window_from_signal_for_RP_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor_start=curr_anchor_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 3: # draw ts anchor1
                    curr_window_type = "anchor1"
                    curr_anchor1_start = None
                    curr_anchor2_start = None
                    curr_tpos = None
                    curr_tneg = None
                    sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor1_start=curr_anchor1_start,
                        anchor2_start=curr_anchor2_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 4: # draw ts anchor2
                    curr_window_type = "anchor2"
                    curr_anchor1_start = input_starts[3]
                    curr_anchor2_start = None
                    curr_tpos = self.task_variable_parameter_args.ts_tpos
                    curr_tneg = None
                    sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor1_start=curr_anchor1_start,
                        anchor2_start=curr_anchor2_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                elif window_num == 5: # draw ts other
                    curr_window_type = "other"
                    curr_anchor1_start = input_starts[3]
                    curr_anchor2_start = input_starts[4]
                    curr_tpos = None
                    curr_tneg = None
                    if labels[2] == BINARY_NEG_LABEL:
                        curr_tneg = self.task_variable_parameter_args.ts_tneg
                    elif (
                        labels[2] == BINARY_POS_LABEL
                    ):  # elif labels[1] == BINARY_POS_LABEL:
                        curr_tpos = self.task_variable_parameter_args.ts_tpos
                    else:
                        raise ValueError(
                            "TUHAbnormalDataset.draw_sample: unsupported label for TS task requested"
                        )

                    sampled_window, start_ind = sample_window_from_signal_for_TS_task(
                        full_signal,
                        self.task_variable_parameter_args.window_len,
                        window_type=curr_window_type,
                        anchor1_start=curr_anchor1_start,
                        anchor2_start=curr_anchor2_start,
                        tpos=curr_tpos,
                        tneg=curr_tneg,
                    )
                    input_starts.append(start_ind)
                else:
                    raise ValueError(
                        "TUHAbnormalDataset.draw_sample: too many windows requested for NonAnchoredBTUABRPTS task"
                    )
            
            else:
                raise NotImplementedError(
                    "TUHAbnormalDataset.draw_sample: window sampling not implemented for self.task_id == "
                    + self.task_id
                )

            # filtered_window = (
            #     mne.filter.filter_data(  # apply a 4th order butterworth filter
            #         data=sampled_window,
            #         sfreq=self.task_variable_parameter_args.fs,
            #         l_freq=self.task_variable_parameter_args.l_freq,
            #         h_freq=self.task_variable_parameter_args.h_freq,
            #         method=self.task_variable_parameter_args.filter_method,
            #         fir_window=self.task_variable_parameter_args.fir_window,
            #     )
            # )
            
            # inputs.append(
            #     filtered_window[:, 0::1]
            # )  # remove the downsample (fs=1000 -> 250Hz downsample) and record as input

            # normalize and transpose window - see https://stackoverflow.com/questions/8717139/how-to-normalize-a-signal-to-zero-mean-and-unit-variance
            unnormalized_shape = sampled_window.shape
            curr_num_chans = unnormalized_shape[0]
            means = np.reshape(np.mean(sampled_window, axis=1), (curr_num_chans,1))
            standard_devs = np.reshape(np.std(sampled_window, axis=1), (curr_num_chans,1))
            sampled_window = (sampled_window - means)/standard_devs
            assert sampled_window.shape == unnormalized_shape
            
            inputs.append(sampled_window.T)

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
        TUHAbnormalDataset.cache_samples_for_current_split_fold: caches dataset for a given cross-validation split
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
            * TUHAbnormalDataset.preprocess_and_cache_data: uses TUHAbnormalDataset.get_number_of_samples_to_draw_from_each_file prior to drawing samples
        """
        source_file_paths = [
            x for x in all_data_file_paths for y in fold_individs if y in x
        ]
        shuffle(source_file_paths)

        num_samps_per_source_file = [
            num_samps_across_files
            // len(
                source_file_paths
            )  # len(source_file_paths) // num_samps_across_files
            for _ in range(len(source_file_paths))
        ]
        for i in range(num_samps_across_files % len(source_file_paths)):
            num_samps_per_source_file[i] += 1

        curr_subset_id_counter = 0
        curr_subset = []
        #idx_dictionary: idx_dict = {idx: [file_name, row_ind]}
        #idx_dict = {0: ["file_path", 0], 1: ["file_path", 1]}
        idx_dictionary = {}
        entire_row_idx_counter = 0

        #curr_subset_save_path is directory/subset/subset_id.npy
        curr_subset_save_path = os.sep.join(
            [fold_save_dir, "subset" + str(curr_subset_id_counter) + ".npy"]
        )
        for file_path, num_samps_needed in zip(
            source_file_paths, num_samps_per_source_file
        ):
            for i in range(num_samps_needed):
                # draw sample
                curr_subset.append(
                    self.draw_sample(file_path[:-4] + self.unique_temp_file_id + ".npy")
                )  # note: assumes .pkl file was saved in prior call to self.filter_data
                # curr_subset has sampled data * num_samps_needed amount
                # check if current subset needs to be cached
                if len(curr_subset) == max_num_samps_per_subset_file:
                    with open(curr_subset_save_path, "wb") as outfile:
                        # pkl.dump(curr_subset, outfile)
                        #file_array = [[each row (eacmple)], []]
                        file_array = np.array(curr_subset)
                        np.save(outfile, file_array)
                        # add each row (different sampled data) in each file into dictionary with index
                        for row_idx_in_file, row in enumerate(file_array): 
                            #idx_dictionary: idx_dict = {idx: [file_name, row_ind]}
                            idx_dictionary[entire_row_idx_counter] = [curr_subset_save_path, row_idx_in_file]
                            entire_row_idx_counter += 1
                        
                    # initialize new subset
                    curr_subset_id_counter += 1
                    curr_subset = []
                    curr_subset_save_path = os.sep.join(
                        [fold_save_dir, "temp_subset" + str(curr_subset_id_counter) + ".npy"]
                    )

        random_seed = 0
        #shuffle keys of rows, where a key is the number of rows starting from 0 for all the subset files)
        random.Random(random_seed).shuffle(list(idx_dictionary.keys()))

        #result_final_arr = [["file_name", row_index, row_array[]]] where row_index is the number of row in that particular file counting from the first row in the file.
        result_final_arr = []
        for idx in idx_dictionary.keys():
            loaded_file = np.load(idx_dictionary[idx][0], allow_pickle=True)
            row_index = idx_dictionary[idx][1]
            curr_row = loaded_file[row_index]
            result_final_arr.append(curr_row)

            if len(result_final_arr) == max_num_samps_per_subset_file:
                curr_subset_save_path = os.sep.join(
                [fold_save_dir, "subset" + str(idx) + ".npy"])
                with open(curr_subset_save_path, "wb") as final_outfile:
                    #result_final_arr includes [[curr_row], [curr_row] ..]
                    # np.save(np.array(result_final_arr), final_outfile)
                    np.save(final_outfile, np.array(result_final_arr))
        print("Total number of samples in subset(number of rows): ", len(idx_dictionary.keys()))

        return source_file_paths, num_samps_per_source_file

    def preprocess_and_cache_data(self, data_source_specific_args):
        """
        - Inputs:
            * data_source_specific_args (argparse obj): arguments for preprocessing the original CPNE Fluoxetine dataset, including
                - args.original_data_source_dir (str): the directory containing the original CPNE Fluoxetine dataset
        """
        # compile a list of good edf files
        print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.preprocess_and_cache_data: NOW FILTERING DATA")
        self.filter_data(data_source_specific_args)

        # get the train and test individual id data
        print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.preprocess_and_cache_data: NOW GETTING INDIVIDUALS")
        (
            individual_ids_train,
            individual_ids_test,
            all_data_file_paths,
        ) = self.get_individuals(data_source_specific_args.data_source_path)

        print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.preprocess_and_cache_data: NOW SPLITTING INDIVIDUALS")
        self.data_splits_by_individual_id = self.split_individual_ids_in_dataset(
            individual_ids_train, individual_ids_test
        )

        [
            self.num_total_train_samples_per_split,
            self.num_total_val_samples_per_split,
            self.num_total_test_samples,
        ] = data_source_specific_args.nums_of_samples_per_split_by_fold
        # self.window_size = data_source_specific_args.window_size
        self.max_num_samps_per_subset_file = (
            data_source_specific_args.max_num_samps_per_subset_file
        )

        print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.preprocess_and_cache_data: NOW CACHING SAMPLES FOR EACH CROSS VALIDATION FOLD")
        holdout_set_has_been_cached = False
        split_subset_info = {}
        curr_test_source_file_paths = None
        curr_test_num_samps_per_source_file = None
        for curr_split_num, [
            train_split_individs,
            val_split_individs,
            test_split_individs,
        ] in enumerate(self.data_splits_by_individual_id):
            # call self.cache_samples_for_current_split_fold for each of train/val/test and store returned values
            (
                curr_train_source_file_paths,
                curr_train_num_samps_per_source_file,
            ) = self.cache_samples_for_current_split_fold(
                train_split_individs,
                self.good_edf_files,
                self.num_total_train_samples_per_split,
                self.data_save_directory + os.sep + "train" + str(curr_split_num),
                self.max_num_samps_per_subset_file,
            )
            (
                curr_val_source_file_paths,
                curr_val_num_samps_per_source_file,
            ) = self.cache_samples_for_current_split_fold(
                val_split_individs,
                self.good_edf_files,
                self.num_total_val_samples_per_split,
                self.data_save_directory + os.sep + "validation" + str(curr_split_num),
                self.max_num_samps_per_subset_file,
            )
            if not holdout_set_has_been_cached:
                (
                    curr_test_source_file_paths,
                    curr_test_num_samps_per_source_file,
                ) = self.cache_samples_for_current_split_fold(
                    test_split_individs,
                    self.good_edf_files,
                    self.num_total_test_samples,
                    self.data_save_directory + os.sep + "test", # + str(curr_split_num),
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

            # # FOR DEBUGGING PURPOSES
            # print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.preprocess_and_cache_data: BREAKING FOR DEBUGGING PURPOSES")
            # break
            pass

        with open(
            self.data_save_directory + os.sep + "cached_data_stats_and_params.pkl", "wb"
        ) as outfile:
            pkl.dump(
                {
                    "all_data_file_paths": all_data_file_paths,
                    "individual_ids_train": individual_ids_train,
                    "individual_ids_test": individual_ids_test,
                    "data_splits_by_individual_id": self.data_splits_by_individual_id,
                    "data_source_specific_args": data_source_specific_args,
                    "split_subset_info": split_subset_info,
                    "task_variable_parameter_args": self.task_variable_parameter_args, # "good_channels": self.good_channels, 
                    "unique_temp_file_id": self.unique_temp_file_id, 
                },
                outfile,
            )
        
        print("preprocess_TUHAbnormal_data.TUHAbnormalDataset.preprocess_and_cache_data: FINISHED CACHING!!")
        pass

        # # all the train and validation file path after split
        # (train_source_file_paths, val_source_file_paths) = self.train_split_path(
        #     individual_ids_train, all_data_file_paths
        # )
        # # all the test file path
        # test_source_file_paths = self.test_path(
        #     individual_ids_test, all_data_file_paths
        # )

        # # get edf file list for train data
        # train_edf_file_list: list[str] = []
        # val_edf_file_list: list[str] = []
        # for train_source_file_path in train_source_file_paths:
        #     train_session_dir = os.listdir(train_source_file_path)
        #     for train_session in train_session_dir:
        #         # print(train_session)
        #         edf_directory_path = os.sep.join(
        #             [train_source_file_path, train_session]
        #         )
        #         edf_file = [x for x in os.listdir(edf_directory_path) if "edf" in x]
        #         edf_file_path = os.sep.join([edf_directory_path, edf_file[0]])
        #         if edf_file_path in self.good_edf_files:
        #             train_edf_file_list.append(edf_file_path)

        # # get edf file list for validation data
        # for val_source_file_path in val_source_file_paths:
        #     val_session_dir = os.listdir(val_source_file_path)
        #     for val_session in val_session_dir:
        #         # print(train_session)
        #         edf_directory_path = os.sep.join([val_source_file_path, val_session])
        #         edf_file = [x for x in os.listdir(edf_directory_path) if "edf" in x]
        #         edf_file_path = os.sep.join([edf_directory_path, edf_file[0]])
        #         if edf_file_path in self.good_edf_files:
        #             val_edf_file_list.append(edf_file_path)

        # for train_edf_file in train_edf_file_list:
        #     signal_data = mne.io.read_raw_edf(train_edf_file)
        #     print(signal_data)
        # [
        #     self.num_total_train_samples_per_split,
        #     self.num_total_val_samples_per_split,
        # ] = data_source_specific_args.nums_of_samples_per_split_by_fold
        # self.window_size = data_source_specific_args.window_size


# if __name__ == "__main__":
#     """
#     __main__: Preprocesses the TUH Abnormal (biosignal) dataset for cross-domain-learning experiments
#      - Inputs:
#         * cached_args_file (str): the name of a cached argument file (.pkl format) from which to pull parameters for TUHAbnormal processing
#         * n (int): a bandaid for running this function in Slurm which is ignored by this code
#      - Outputs:
#         * *Split_*_TUHAbnormal_data_*/ (preprocessed data directory): Preprocessed TUH Abnormal data is saved to the given
#             data_save_dir. See cross-domain-learning/file_directory_templates.txt for details.
#      - Usage:
#         * run from terminal/bash scripts: this is an end-point function whose sole-puropose is to be run
#     """

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-cached_args_file",
#         type=str,
#         default="cached_args_preprocess_TUHAbnormal_data.pkl",
#         help="/path/to/cached_args.pkl should contain an argparse.ArgumentParser() with arguments structured as in the __main__ doc string",
#     )
#     parser.add_argument(
#         "-n", default=None, help="IGNORE THIS ARGUMENT - IT'S A BANDAID"
#     )

#     args = parser.parse_args()

#     print("<<< MAIN: START")
#     preprocess = PreprocessedTUHAbnormalDataset(
#         root_save_directory="/Users/akira/CodeProjects/cdl",
#         data_split_type="crossValSplit",
#         task_id="TS",
#         data_source_id="TUHAbnormal",
#         date_of_data_split="YYYYMMDD",
#     )
#     preprocess.preprocess_and_cache_data("/Users/akira/CodeProjects/cdl/edf")
#     # with open(args.cached_args_file, "rb") as infile:
#     #     cached_args = pkl.load(infile)
#     #     print("<<< cached_args == ", cached_args)
#     #     args.data_save_dir = cached_args.data_save_dir
#     #     print("<<< new args == ", args)

#     # taskID = set_up_and_run_current_experiment(args)

#     # print("<<< MAIN: DONE RUNNING TASKID == ", taskID, "!!!")
#     pass

# """
# --- Cross-Val Training Dataset Directory with n Train-Val-Test Splits: ---
# Example: "/crossValSplit_TS_TUHAbnormal_data_20210920/train2/train_subset0.pkl"
# 1. Root Dataset Save Directory Name
#  * Folder name structure:
#     - 1st element: "crossValSplit_"; identifies the directory as containing n train-val splits and (optionally) 1 holdout test set
#     - 2nd element: task_id_str+"_"; a descriptive string identifying the task for which the dataset was currated (e.g. "cdisnRP", "TS", "MYOWMined", etc)
#     - 3rd element: data_source_str+"_"; a descriptive string identifying the source of the dataset (e.g. "TUHAbnormal")
#     - 4th element: "data_"+date_str; date_str states the date the data-split was generated, formatted as YYYYMMDD
#  * Contents:
#     - Subdirectories: train1/, validation1/, ..., trainN/, validationN/, and (optionally) test/
#     - Other:
#        * ftion.pkl: a file containing a python dict object storing info such as
#           - "source_info": where the dataset was sourced from
#           - "config_args": the arguments given to the preprocessing pipeline when the split was generated
#           - other info as needed
# 2. Train-Val-Test Subdirectory Name
#  * Folder name structure: one of 'trainX/', 'validationX/', or 'test/'
#  * Contents:
#     - Subset Files: .pkl files containing a subset (i.e. multiple batches) of the train/val/test set to iterate over

# """

if __name__ == "__main__":
    """
    preprocess_TUHAbnormal_data.__main__: shuffles, splits, and caches TUAB data based on provided arguments
    - Inputs:
       * cached_args_file (argparse obj pickle file): arguments for preprocessing the original TUAB dataset, including
          - cached_args.root_save_directory (str): the directory containing the original TUAB dataset
          - cached_args.data_split_type (str): one of ['crossValSplit'] denoting the type of dataset being curated
          - cached_args.task_id (str): one of the id's present in self.get_hyperparams_for_task cases denoting the types of tasks being modeled in the current dataset
          - cached_args.data_source_id (str): an identifier of the original dataset (e.g., 'TUAB' or 'CPNEFluoxetine')
          - cached_args.date_of_data_split (str): a YYYYMMDD-formated string denoting the date of dataset curation/preprocessing
          - cached_args.task_variable_parameter_args (argparse obj): taski-specific hyperparameters (e.g., tau-pos for RP task) which may need tuning, default=None,
          - cached_args.data_source_specific_args (argparse obj): arguments for preprocessing the TUAB dataset, default=None,
          - cached_args.num_splits (int): the number of train-val-test splits to make (esp. for cross-validation), default=1, # - cached_args.holdout_set (boolean): whether a test split needs to be stored as a holdout set, default=False,
          - cached_args.split_ratios (tuple): a tuple containing the ratios (float) of each split data subset to be assigned to the train, validation, and test sets, respectively, default=(0.7, 0.2, 0.1),
    - Outputs:
       * */cached_samples*.pkl (cached list): pickle files containing lists of cached samples, with each sample formatted as (x1, x2, ..., xn, label_y)
    - Usage:
       * PreprocessedDataset.__init__: uses TUHAbnormalDataset.preprocess_and_cache_data when self.data_save_directory is empty
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cached_args_file",
        type=str,
        default="cached_args_preprocess_TUHAbnormal_data.pkl",
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

    with open(
        cached_args.root_save_directory + os.sep + "cached_data_preprocessing_args.pkl",
        "wb",
    ) as outfile:
        pkl.dump(cached_args, outfile)

    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print("<<< KICKING OFF TASKID == ", taskID)
    _ = TUHAbnormalDataset(
        cached_args.root_save_directory,
        cached_args.data_split_type,
        cached_args.task_id,
        cached_args.data_source_id,
        cached_args.date_of_data_split,
        cached_args.task_variable_parameter_args,
        cached_args.data_source_specific_args,
        cached_args.num_splits,  # cached_args.holdout_set,
        cached_args.split_ratios, # cached_args.num_edf_filtering_samples, # cached_args.max_num_bad_edf_filtering_samples,
    )

    print("<<< MAIN: DONE RUNNING TASKID == ", taskID, "!!!")
    pass

"""
Args to delete:
max_num_bad_edf_filtering_samples
num_edf_filtering_samples
task_variable_parameter_args.filter_method
task_variable_parameter_args.fir_window
task_variable_parameter_args.fs
task_variable_parameter_args.h_freq
task_variable_parameter_args.l_freq
-----
Args to change
task_variable_parameter_args.window_len = 600
"""

