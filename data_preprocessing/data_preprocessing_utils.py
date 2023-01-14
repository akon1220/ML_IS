# data_preprocessing_utils.py: a file defining general data_preprocessing classes and functions for the cross-domain-learning repository
# SEE LICENSE STATEMENT AT THE END OF THE FILE

# dependency import statements
import os
import pickle as pkl
import torch
import random
from random import seed as randseed
from random import shuffle
from random import choice as randchoice
import numpy as np
from sklearn.model_selection import GroupKFold

from utils.caching_utils import create_directory
from utils import create_directory



class CachedDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()  # see https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        self.INDEX_MAP_SAMPLE_PATH_LOC = 0
        self.INDEX_MAP_SAMPLE_SUBINDEX_LOC = 1
        self.num_samples_in_curr_subset = None
        self.curr_subset_sample_index_map = None
        pass

    def load_subset(self, subset_path, max_data_samp_ratio=None, random_seed=0):
        """
        CachedDataset.load_subset: loads dataset subset from cached file an populates self.curr_subset_sample_index_map, self.current_subset_path, and self.num_samples_in_curr_subset accordingly
         - Inputs:
            * subset_path (str): the path of the dataset subset to be loaded
         - Outputs:
            * N/A
         - Usage:
            * PreprocessedDataset.__init__: uses CachedDataset.load_subset to during PreprocessedDataset instance setup
        """
        print("CachedDataset.load_subset: Original subset path == ", subset_path)
        curr_subset_sample_index_map = {} 
        self.current_subset_path = subset_path
        # cached_sample_file_names = os.listdir(subset_path)
            #self.data should be a numpy array - including [[curr_row], [curr_row]] * number of rows
        with open(subset_path + ".npy", "rb") as infile: 
            self.data = np.load(infile)
        #number of rows is number of samples in the current subset

        if max_data_samp_ratio is not None: 
            #TODO: decrease the size of self.data (less number of rows) CHECK if this is correct. do we need to shuffle?
            num_samples_to_keep = int(self.data.shape[0]*max_data_samp_ratio)
            # random.Random(random_seed).shuffle()
            self.data = self.data[:num_samples_to_keep]

        self.num_samples_in_curr_subset = self.data.shape[0]

        print("CachedDataset.load_subset: Original number of samples in subset == ", self.num_samples_in_curr_subset)



        #this number is same as curr_index_counter
        # if max_data_samp_ratio is not None:
        #     reduced_subset_sample_index_map = {}
        #     curr_sample_indices = list(curr_subset_sample_index_map.keys())
        #     assert len(curr_sample_indices) == len(curr_subset_sample_index_map.keys()) # sanity check
        #     num_samples_to_keep = int(len(curr_sample_indices)*max_data_samp_ratio)
        #     random.Random(random_seed).shuffle(curr_sample_indices) # see https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
        #     for new_ind, old_ind in enumerate(curr_sample_indices[:num_samples_to_keep]):
        #         reduced_subset_sample_index_map[new_ind] = curr_subset_sample_index_map[old_ind]
        #     curr_subset_sample_index_map = reduced_subset_sample_index_map

        # self.curr_subset_sample_index_map = curr_subset_sample_index_map
        print("CachedDataset.load_subset: Final number of samples in subset == ", self.num_samples_in_curr_subset)
        # assert self.num_samples_in_curr_subset == curr_index_counter  # sanity check



    def __len__(self):
        """
        PreprocessedDataset.__len__: returns the number of samples stored in cached dataset
         - Inputs:
            * N/A
         - Outputs:
            * self.num_samples_in_curr_subset (int): the number of samples stored in currently loaded dataset subset (numbers of rows in matrix)
         - Usage:
            * N/A
        """
        return self.num_samples_in_curr_subset

    def __getitem__(self, index):
        """
        PreprocessedDataset.__getitem__: returns the data sample referenced by index in self.curr_subset_sample_index_map
         - Inputs:
            * index (int): the index of the sample in loaded dataset subset being requested
         - Outputs:
            * curr_sample (tuple): curr sample is assumed to be a sample tuple formatted as (x1, x2, ..., xn, label_y)
         - Usage:
            * N/A
        """
        #just return a index-th row of the matrix
        return self.data[index]


class CachedDatasetForTraining(CachedDataset):
    def __init__(
        self,
        root_save_directory,
        subset_foldername,
        max_data_samp_ratio=None, 
    ):
        super().__init__()  # see https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        # super(CachedDataset, self).__init__()

        print(
            "CachedDatasetForTraining.__init__: attempting to load cached data from existing data_save_directory"
        )
        self.current_subset_path = root_save_directory + os.sep + subset_foldername
        self.max_data_samp_ratio = max_data_samp_ratio
        
        pass

    pass


class PreprocessedDataset(CachedDataset):
    """
    PreprocessedDataset Description:
    - General Purpose: defines a parent class for all datasets used in cross-domain-learning repository
    - Usage:
      * data_preprocessing_utils.preprocess_*_data.py: uses PreprocessedDataset children to standardize processed data directories for various tasks
      * data_loading_utils.load_*_data.py: uses PreprocessedDataset children to load processed data for various tasks
    """

    def __init__(
        self,
        root_save_directory,
        data_split_type,
        task_id,
        data_source_id,
        date_of_data_split,  # num_inputs_per_sample=None, num_labels_per_sample=None, num_classes=None, input_augmentations=None,
        task_variable_parameter_args=None,
        data_source_specific_args=None,
        num_splits=1,
        split_ratios=(
            0.7,
            0.2,
            0.1,
        ),  # split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
        max_data_samp_ratio=None, 
    ):
        """
        PreprocessedDataset.__init__: defines directory, filenames into which preprocessed data will be saved in addition to member variables
         - Inputs:
            * root_save_directory (str): the (root) directory to which the preprocessed data will be saved - see templates.file_directory_templates for details
            * data_split_type (str): one of ['crossValSplit'] denoting the type of dataset being curated
            * task_id (str): one of the id's present in self.get_hyperparams_for_task cases denoting the types of tasks being modeled in the current dataset
            * data_source_id (str): an identifier of the original dataset (e.g., 'TUAB' or 'CPNEFluoxetine')
            * date_of_data_split (str): a YYYYMMDD-formated string denoting the date of dataset curation/preprocessing
            * task_variable_parameter_args (argparse obj): arguments for task-specific parameters (e.g., tau-positive for RP task) which we may wish to tune in experiments
            * data_source_specific_args (argparse obj): arguments for preprocessing a specific original dataset (e.g., 'TUAB') - OVERWRITTEN BY CHILD CLASSES, default=None
            * num_splits (int): the number of train-val-test splits to make (esp. for cross-validation), default=1
            * split_ratios (tuple): a tuple containing the ratios (float) of each split data subset to be assigned to the train, validation, and test sets, respectively, default=(0.7, 0.2, 0.1)
         - Outputs:
            * N/A
         - Usage:
            * PreprocessedDataset class: uses PreprocessedDataset.__init__ to initialize member variables
            * CPNEFluoxetineDataset.__init__: calls PreprocessedDataset.__init__ to initialize member variables and perform preprocessing/caching operations
        """
        super().__init__()  # see https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        # super(CachedDataset, self).__init__()
        self.NUM_SUBSTRS_IN_ROOT_FOLDER_NAME = 5
        self.DATASPLIT_TYPE_SUBSTR_LOC = 0
        self.TASK_ID_SUBSTR_LOC = 1
        self.DATA_SOURCE_SUBSTR_LOC = 2
        self.DATE_SUBSTR_LOC = 4
        self.TRAIN_RATIO_IND = 0
        self.VAL_RATIO_IND = 1
        self.TEST_RATIO_IND = 2

        # perform checks on input args
        assert data_split_type in ["crossValSplit"]
        assert num_splits >= 1
        assert (
            sum(list(split_ratios)) - 0.0001 < 1.0
            and sum(list(split_ratios)) + 0.0001 > 1.0
        )

        # store configuration info for root_folder_name
        self.data_split_type = data_split_type
        self.task_id = task_id
        self.data_source_id = data_source_id
        self.date_of_data_split = date_of_data_split
        self.split_ratios = split_ratios
        self.num_splits = num_splits

        # get constant task-specific hyperparameters for the task modeled by the current dataset
        (
            num_inputs_per_sample,
            num_labels_per_sample,
            num_classes,
            input_augmentations,
        ) = self.get_hyperparams_for_task()

        self.num_inputs_per_sample = num_inputs_per_sample
        self.num_labels_per_sample = num_labels_per_sample
        self.num_classes = num_classes
        self.input_augmentations = input_augmentations
        self.task_variable_parameter_args = task_variable_parameter_args

        if self.task_variable_parameter_args.throttle_ratio is not None:
            self.throttle_percent = max(
                int(self.task_variable_parameter_args.throttle_ratio * 100), 1
            )
        else:
            self.throttle_percent = 100

        # begin creating data_save_directory path - see templates.file_directory_templates
        root_folder_name = "_".join(
            [
                data_split_type,
                task_id,
                data_source_id,
                str(self.throttle_percent) + "percentOfData",
                date_of_data_split,
            ]
        )
        self.data_save_directory = root_save_directory + os.sep + root_folder_name
        self.max_data_samp_ratio = max_data_samp_ratio
        self.current_subset_path = self.data_save_directory + os.sep + "train0"

        # initialize dataset member variables for retrieving data samples
        self.curr_subset_sample_index_map = None
        # will be overwritten by child-class methods, probably to look like {dataset_index_int: (file_path_str, file_index_int)}
        self.num_samples_in_curr_subset = 0

        if os.path.exists(
            self.data_save_directory
        ):  # check for existance of data_save_directory and load cached data if it exists
            print(
                "PreprocessedDataset.__init__: attempting to load cached data from existing data_save_directory"
            )
            self.load_subset(self.current_subset_path, self.max_data_samp_ratio)
        else:  # create data_save_directory and populate accordingly
            create_directory(self.data_save_directory)

            # initialize subdirectories, depending on split-type
            for i in range(num_splits):
                create_directory(self.data_save_directory + os.sep + "train" + str(i))
                create_directory(
                    self.data_save_directory + os.sep + "validation" + str(i)
                )
            if self.split_ratios[-1] > 0.0 or "TUAB" in self.task_id:
                create_directory(self.data_save_directory + os.sep + "test")

            print("Finished making directories!!")
            # populate subdirectories: requires shuffling / splitting / batching of data
            self.preprocess_and_cache_data(data_source_specific_args)

        # save what we can - esp. dataset_configuration.pkl
        with open(
            self.data_save_directory + os.sep + "dataset_configuration.pkl", "wb"
        ) as outfile:
            pkl.dump(
                {
                    "root_save_directory": root_save_directory,
                    "data_split_type": data_split_type,
                    "task_id": task_id,
                    "data_source_id": data_source_id,
                    "date_of_data_split": date_of_data_split,
                    "task_variable_parameter_args": task_variable_parameter_args,
                    "data_source_specific_args": data_source_specific_args,
                    "num_splits": num_splits,
                    "split_ratios": split_ratios,
                    "current_subset_path": self.current_subset_path,
                    "curr_subset_sample_index_map": self.curr_subset_sample_index_map,
                    "num_samples_in_curr_subset": self.num_samples_in_curr_subset,
                    "throttle_percent": self.throttle_percent,
                },
                outfile,
            )
        pass

    def get_hyperparams_for_task(self):
        """
        PreprocessedDataset.get_hyperparams_for_task: returns task-specific hyperparameters for the task modeled by the current dataset
         - Inputs:
            * N/A
         - Outputs:
            * num_inputs_per_sample (int): the number of inputs to assign to each sample (e.g., recording windows)
            * num_labels_per_sample (int): the number of labels to assign to each sample
            * num_classes (int): the number of classes represented by each label in the dataset
            * input_augmentations (list): list of different augmentations to apply to each sample in the dataset
         - Usage:
            * PreprocessedDataset.__init__: uses PreprocessedDataset.get_hyperparams_for_task to determine how to format each sample in preprocessed dataset
        """
        # initialize task-specific variables
        num_inputs_per_sample = None
        num_labels_per_sample = None
        num_classes = None
        input_augmentations = None
        if self.task_id == "RP":  # see arxiv.org/pdf/2007.16104.pdf
            num_inputs_per_sample = 2  # anchor and sample window
            num_labels_per_sample = 1  # hard label
            num_classes = 2  # positive or negative context
            # input_augmentations = None
        elif self.task_id == "TS":  # see arxiv.org/pdf/2007.16104.pdf
            num_inputs_per_sample = 3  # 2 anchors and sample window
            num_labels_per_sample = 1  # hard label
            num_classes = 2  # positive or negative context
            # input_augmentations = None
        elif self.task_id == "BehavioralTST":
            num_inputs_per_sample = 1  # a sample window
            num_labels_per_sample = 1  # hard label
            num_classes = 3  # home cage, open field, and tail-suspension
            # input_augmentations = None
        elif self.task_id == "BehavioralFluoxetine":
            num_inputs_per_sample = 1  # a sample window
            num_labels_per_sample = 1  # hard label
            num_classes = 4  # home cage, open field, saline, and fluoxetine
            # input_augmentations = None
        elif self.task_id == "BehavioralTUAB":
            num_inputs_per_sample = 1  # a sample window
            num_labels_per_sample = 1  # hard label
            num_classes = 2  # normal and abnormal
            # input_augmentations = None
        elif self.task_id == "AnchoredBTUABRPTS":
            num_inputs_per_sample = 4  # an anchor, rp other, ts anchor2, and ts other window
            num_labels_per_sample = 3  # soft labels
            num_classes = None  # not applicable due to soft labels
            # input_augmentations = None
        elif self.task_id == "NonAnchoredBTUABRPTS":
            num_inputs_per_sample = 6  # a sample window, 2 rp windows, and 3 ts windows
            num_labels_per_sample = 3  # soft labels
            num_classes =  None  # not applicable due to soft labels
            # input_augmentations = None
        else:
            raise ValueError(
                "PreprocessedDataset.get_hyperparams_for_task: The following task id is not supported - "
                + str(self.task_id)
            )

        return (
            num_inputs_per_sample,
            num_labels_per_sample,
            num_classes,
            input_augmentations,
        )

    def restrict_num_individs_by_throttle_setting(self, individual_ids):
        throttle_ratio = self.task_variable_parameter_args.throttle_ratio
        num_ids_to_use = max(int(throttle_ratio * len(individual_ids)), 1)
        shuffle(individual_ids)
        return individual_ids[:num_ids_to_use]
    
    def get_num_ids_per_split_subset(self, ids, split_ratios):
        num_ids_per_split_subset = []
        for ratio in list(split_ratios):
            if ratio is not None:
                num_ids_per_split_subset.append(int(ratio * len(ids)))
        assert sum(num_ids_per_split_subset) <= len(ids)  # sanity check

        if sum(num_ids_per_split_subset) != len(ids):  # account for rounding errors
            if (
                num_ids_per_split_subset[-1] != 0
            ):  # case in which we append leftover ids to test set
                num_ids_per_split_subset[-1] = num_ids_per_split_subset[-1] + (
                    len(ids) - sum(num_ids_per_split_subset)
                )
            else:  # case in which we append leftover ids to validation set
                num_ids_per_split_subset[-2] = num_ids_per_split_subset[-2] + (
                    len(ids) - sum(num_ids_per_split_subset)
                )
        return num_ids_per_split_subset
        pass

    def split_individual_ids_in_dataset(self, ids, test_ids=None):
        """
        PreprocessedDataset.split_individual_ids_in_dataset: returns cross-validation (and holdout-set) splits according to individual id in dataset
         - Inputs:
            * ids (list): a list of unique ids corresponding to all of the individuals (i.e., animals for CPNEFluoxetine, people for TUAB, etc) represented in dataset
         - Outputs:
            * id_splits (list(list)): splits of the dataset by individual ids, formatted as [[train1_ids,val1_ids,test_ids], ...] with *_ids==[id1, id2, id3,...]
         - Usage:
            * PreprocessedDataset.preprocess_and_cache_data: uses PreprocessedDataset.split_individual_ids_in_dataset to determine which individuals to put into each data split
        """
        if "TUAB" in self.task_id: # == "BehavioralTUAB": # sanity check
            assert self.split_ratios[-1] == 0.0

        shuffle(
            ids
        )  # see https://stackoverflow.com/questions/976882/shuffling-a-list-of-objects

        # determine how many ids to put into each train/validation/(test) split
        num_ids_per_split_subset = self.get_num_ids_per_split_subset(ids, self.split_ratios)

        # make splits
        # test_ids = None
        if test_ids is None and self.split_ratios[-1] > 0.0:
            test_ids = ids[len(ids) - num_ids_per_split_subset[2] :]
            assert len(test_ids) == num_ids_per_split_subset[2]
            ids = ids[: len(ids) - num_ids_per_split_subset[2]]
            if self.num_splits >= 2:
                num_ids_per_split_subset = self.get_num_ids_per_split_subset(ids, (1.-self.split_ratios[1], self.split_ratios[1], 0.0))

        id_splits = []
        if (
            self.num_splits >= 2
        ):  # see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
            groups = [i for i in range(len(ids))]
            group_kfold = GroupKFold(n_splits=self.num_splits)
            group_kfold.get_n_splits(
                np.array(groups), np.array(groups), np.array(groups)
            )
            for _, val_index in group_kfold.split(
                np.array(groups), np.array(groups), np.array(groups)
            ):
                val_ids = []
                for ind in val_index:
                    val_ids.append(ids[ind])
                train_ids = [x for x in ids if x not in val_ids]
                # perform sanity checks
                # assert len(train_ids) == num_ids_per_split_subset[0]
                # if not len(train_ids) == num_ids_per_split_subset[0]:
                #     print("UNEXPECTED NUMBER OF train_ids ENCOUNTERED: len(train_ids)==", len(train_ids), " whereas num_ids_per_split_subset == ", num_ids_per_split_subset)
                #     raise ValueError("UNEXPECTED NUMBER OF train_ids ENCOUNTERED")
                # # assert len(val_ids) == num_ids_per_split_subset[1]
                # if not len(val_ids) == num_ids_per_split_subset[1]:
                #     print("UNEXPECTED NUMBER OF val_ids ENCOUNTERED: len(val_ids)==", len(val_ids), " whereas num_ids_per_split_subset == ", num_ids_per_split_subset)
                #     raise ValueError("UNEXPECTED NUMBER OF val_ids ENCOUNTERED")
                # # assert sorted(train_ids + val_ids) == sorted(ids)
                # if not sorted(train_ids + val_ids) == sorted(ids):
                #     print("UNEXPECTED NUMBER OF ids IN SPLIT: len(train_ids)==", len(train_ids), " len(val_ids)==", len(val_ids), " len(ids)==", len(ids))
                #     raise ValueError("UNEXPECTED NUMBER OF ids IN FINAL SPLIT ENCOUNTERED")
                restricted_train_ids = train_ids
                if self.task_variable_parameter_args.throttle_ratio is not None:
                    restricted_train_ids = (
                        self.restrict_num_individs_by_throttle_setting(
                            restricted_train_ids
                        )
                    )
                # id_splits.append([train_ids, val_ids, test_ids])
                id_splits.append([restricted_train_ids, val_ids, test_ids])
        else:
            train_ids = ids[: num_ids_per_split_subset[0]]
            val_ids = ids[
                num_ids_per_split_subset[0] : num_ids_per_split_subset[0]
                + num_ids_per_split_subset[1]
            ]
            # perform sanity checks
            assert len(train_ids) == num_ids_per_split_subset[0]
            assert len(val_ids) == num_ids_per_split_subset[1]
            # assert sorted(train_ids + val_ids) == sorted(ids)
            restricted_train_ids = train_ids
            if self.task_variable_parameter_args.throttle_ratio is not None:
                restricted_train_ids = self.restrict_num_individs_by_throttle_setting(
                    restricted_train_ids
                )
            # id_splits.append([train_ids, val_ids, test_ids])
            id_splits.append([restricted_train_ids, val_ids, test_ids])

        return id_splits

    def preprocess_and_cache_data(self, data_source_specific_args):
        """
        PreprocessedDataset.load_cached_preprocessed_dataset: reads cached dataset info from self.data_save_directory and stores it in member variables
         - Inputs:
            * data_source_specific_args (argparse obj): arguments for preprocessing a specific original dataset (e.g., 'TUAB') - OVERWRITTEN BY CHILD CLASSES
         - Outputs:
            * N/A
         - Usage:
            * PreprocessedDataset.__init__: uses PreprocessedDataset.preprocess_and_cache_data when self.data_save_directory does not already exist

        !Notes!
         - This function is meant to be inhereted/overwritten by children dataset classes (RPPreprocessedDataset, TSPreprocessedDataset, etc) depending
           on the particular format of the source dataset being preprocessed
        """
        # access source data set and iterate over files
        # open source directory
        # loop through source directory / determine the size that the final preprocessed dataset will be
        # make indices for each sample in the dataset (without storing the samples yet) - indices will likely need to map to a file / location for later access
        # shuffle the sample indices (and mapped file/locations)
        # iterate through shuffled indices, create the sample at current index via task params, and save resulting sample to a small file with data_source_specific_args.num_samples_per_cached_file
        raise NotImplementedError(
            "PreprocessedDataset.preprocess_and_cache_data has not been implemented / overwritten by children class(es)"
        )
        pass


def randomly_sample_window_from_signal(original_signal, window_len, random_seed_val=None):
    """
    data_preprocessing_utils.randomly_sample_window_from_signal: returns the data sample referenced by index in self.curr_subset_sample_index_map
        - Inputs:
            * original_signal (numpy ndarray): an array of shape (num_channels, time) representing the signal from which to draw a sample
            * window_len (int): the length of the window to be sampled
        - Outputs:
            * sampled_window (numpy ndarray): an array of shape (num_channels, window_len) representing the sampled window
            * start_ind (int): the index of the beginning of the sampled window in the original_signal
        - Usage:
            * data_preprocessing_utils.sample_window_from_signal_for_RP_task: uses data_preprocessing_utils.randomly_sample_window_from_signal to draw windows
    """
    signal_len = original_signal.shape[1]  # it is assumed that axis 1 is temporal axis
    available_start_indices = [i for i in range(signal_len - window_len)]
    if random_seed_val is not None:
        # see https://stackoverflow.com/questions/64276987/how-to-fix-the-seed-while-using-random-choice
        randseed(random_seed_val)
    # see https://stackoverflow.com/questions/306400/how-can-i-randomly-select-an-item-from-a-list
    start_ind = randchoice(available_start_indices)
    return original_signal[:, start_ind : start_ind + window_len], start_ind


def sample_window_from_signal_for_RP_task(
    original_signal,
    window_len,
    window_type="anchor",
    anchor_start=None,
    tpos=None,
    tneg=None,
):
    """
    data_preprocessing_utils.sample_window_from_signal_for_RP_task: returns window for use as a portion of a sample input for the RP task
        - Inputs:
            * original_signal (numpy ndarray): an array of shape (num_channels, time) representing the signal from which to draw a sample
            * window_len (int): the length of the window to be sampled
            * window_type (str): the type of window to sample, one of ["anchor", "other"]
            * tpos (int): the size of the context from which to draw positive samples, default=None
            * tneg (int): the size of the context outside of which to draw negative samples, default=None
        - Outputs:
            * sampled_window (numpy ndarray): an array of shape (num_channels, window_len) representing the sampled window
            * start_ind (int): the index of the beginning of the sampled window in the original_signal
        - Usage:
            * N/A
    """
    TEMPORAL_AXIS = 1
    if window_type == "anchor":
        return randomly_sample_window_from_signal(original_signal, window_len)

    original_signal_len = original_signal.shape[
        TEMPORAL_AXIS
    ]  # it is assumed that axis 1 is temporal axis
    assert anchor_start >= 0
    assert anchor_start < original_signal_len

    if tpos is None:
        assert tneg is not None

        buffer_around_anchor = ((tneg // 2) + (tneg % 2)) - (
            (window_len // 2) + (window_len % 2)
        )
        tneg_preceding_buffer = None
        if (
            anchor_start - buffer_around_anchor > window_len
        ):  # check that you can draw a window from preceding negative context
            tneg_preceding_buffer = original_signal[
                :, : anchor_start - buffer_around_anchor
            ]
        tneg_following_buffer = None
        if (
            anchor_start + window_len + buffer_around_anchor
            < original_signal_len - window_len
        ):  # check that you can draw a window from following negative context
            tneg_following_buffer = original_signal[
                :, anchor_start + window_len + buffer_around_anchor :
            ]

        if tneg_preceding_buffer is None and tneg_following_buffer is None:
            raise ValueError(
                "data_preprocessing_utils.sample_window_from_signal_for_RP_task: cannot draw negative sample from provided signal/anchor pair. Consider revising provided tneg and window_len."
            )
        elif tneg_preceding_buffer is not None and tneg_following_buffer is not None:
            portion_of_signal_to_sample = randchoice(
                [tneg_preceding_buffer, tneg_following_buffer]
            )
            return randomly_sample_window_from_signal(
                portion_of_signal_to_sample, window_len
            )
        elif tneg_preceding_buffer is None:
            return randomly_sample_window_from_signal(tneg_following_buffer, window_len)
        else: # tneg_following_buffer is None:
            return randomly_sample_window_from_signal(tneg_preceding_buffer, window_len)

    elif tneg is None:
        assert tpos is not None

        # buffer_around_anchor = ((tpos // 2) + (tpos % 2)) - (
        #     (window_len // 2) + (window_len % 2)
        # )
        buffer_around_anchor = ((tpos - window_len) // 2) + ((tpos - window_len) % 2)
        tpos_preceding_anchor = None
        if (
            buffer_around_anchor >= window_len
            and anchor_start - buffer_around_anchor >= 0
        ):
            tpos_preceding_anchor = original_signal[
                :, anchor_start - buffer_around_anchor : anchor_start
            ]
        tpos_following_anchor = None
        if (
            buffer_around_anchor >= window_len
            and anchor_start + window_len + buffer_around_anchor < original_signal_len
        ):
            tpos_following_anchor = original_signal[
                :, anchor_start + window_len : anchor_start + window_len + buffer_around_anchor
            ]

        if tpos_preceding_anchor is None and tpos_following_anchor is None:
            raise ValueError(
                "data_preprocessing_utils.sample_window_from_signal_for_RP_task: cannot draw positive sample from provided signal/anchor pair. Consider revising provided tpos and window_len."
            )
        elif tpos_preceding_anchor is not None and tpos_following_anchor is not None:
            portion_of_signal_to_sample = randchoice(
                [tpos_preceding_anchor, tpos_following_anchor]
            )
            return randomly_sample_window_from_signal(
                portion_of_signal_to_sample, window_len
            )
        elif tpos_preceding_anchor is None:
            return randomly_sample_window_from_signal(tpos_following_anchor, window_len)
        else: # tpos_following_anchor is None:
            return randomly_sample_window_from_signal(tpos_preceding_anchor, window_len)

    else:
        raise ValueError(
            "data_preprocessing_utils.sample_window_from_signal_for_RP_task: for window_type=="
            + window_type
            + ", tpos or tneg must not be None."
        )
    pass


def sample_window_from_signal_for_TS_task(
    original_signal,
    window_len,
    window_type=None,
    anchor1_start=None,
    anchor2_start=None,
    tpos=None,
    tneg=None,
):
    """
    data_preprocessing_utils.sample_window_from_signal_for_TS_task: returns window for use as a portion of a sample input for the TS task
        - Inputs:
            * original_signal (numpy ndarray): an array of shape (num_channels, time) representing the signal from which to draw a sample
            * window_len (int): the length of the window to be sampled
            * window_type (str): the type of window to sample, one of ["anchor1", "anchor2", "other"]
            * tpos (int): the size (duration) of the context from which to draw positive samples, default=None
            * tneg (int): the size (duration) of the context outside of which to draw negative samples, default=None
        - Outputs:
            * sampled_window (numpy ndarray): an array of shape (num_channels, window_len) representing the sampled window
            * start_ind (int): the index of the beginning of the sampled window in the original_signal
        - Usage:
            * N/A
    """
    if window_type == "anchor1":
        # fix the first anchor position as the beginning of anchor_start to the length of window
        # first_anchor_window = original_signal[:, first_anchor_start:first_anchor_end]
        # return first_anchor_window
        return randomly_sample_window_from_signal(
            original_signal[:, : -3 * window_len], window_len # Note: 2 * window_len may cause an empty list to be available when sampling the 'other' window (? - 12/02/2021)
        )

    TEMPORAL_AXIS = 1

    # get the time from original signal
    original_signal_len = original_signal.shape[TEMPORAL_AXIS]
    # make sure that the any anchor window should be inside the signal.
    assert anchor1_start >= 0
    assert anchor1_start < original_signal_len - (3*window_len) # Note: 2 * window_len may cause an empty list to be available when sampling the 'other' window (? - 12/02/2021)
    if window_type == "other":
        assert anchor2_start is not None
        assert anchor2_start < original_signal_len - window_len
        assert anchor2_start > 2*window_len

    # common variables
    first_anchor_start = anchor1_start
    first_anchor_end = anchor1_start + window_len
    second_anchor_start = first_anchor_start + 3 * window_len # Note: 2 * window_len may cause an empty list to be available when sampling the 'other' window (? - 12/02/2021)

    if window_type == "anchor2":
        tpos_end = first_anchor_start + tpos
        # get the second anchor window. second anchor can be somewhere between second_anchor_start and tpos_end
        portion_of_second_anchor = original_signal[:, second_anchor_start:tpos_end]
        (
            second_anchor_window,
            got_second_anchor_window_start,
        ) = randomly_sample_window_from_signal(portion_of_second_anchor, window_len)
        # initialization of the start of second window
        second_anchor_window_start = second_anchor_start + got_second_anchor_window_start

        return second_anchor_window, second_anchor_window_start

    elif window_type == "other":
        # Tpos implementation (y = 1)
        if tneg == None:
            assert tpos != None
            tpos_end = first_anchor_start + tpos
            # label = 1

            sampled_portion = original_signal[:, first_anchor_end:anchor2_start]
            if sampled_portion.shape[0] == 0:
                raise ValueError("TS sampling function attempted to sample from 0-length portion of signal. first_anchor_end=="+str(first_anchor_end)+" and anchor2_start=="+str(anchor2_start))
            sampled_window, start_ind = randomly_sample_window_from_signal(
                sampled_portion, window_len
            )

            # return (sampled_window, label)
            return sampled_window, first_anchor_end+start_ind

        # Tneg implementation using both tneg and tpos (y = -1)
        else:
            # label = 0
            # the end of second window = start + window_length
            second_anchor_window_end = anchor2_start + window_len

            middle_of_first_second_windows = (
                first_anchor_start + second_anchor_window_end
            ) // 2

            assert (
                first_anchor_start < middle_of_first_second_windows
                and middle_of_first_second_windows < second_anchor_window_end
            )

            half_tneg = (tneg // 2) + (tneg % 2)

            tneg_start = middle_of_first_second_windows - half_tneg
            tneg_end = middle_of_first_second_windows + half_tneg

            assert (
                tneg_start < first_anchor_start and tneg_end > second_anchor_window_end
            )
            # get a sampled window either in the first half of the area outside Tneg or in the second half of the area outside Tneg
            first_sampled_portion = original_signal[:, :tneg_start]
            second_sampled_portion = original_signal[:, tneg_end:]

            if tneg_start <= window_len:
                sampled_window, start_ind = randomly_sample_window_from_signal(
                    second_sampled_portion, window_len
                )
                start_ind = start_ind + tneg_end
            elif original_signal_len - tneg_end <= window_len:
                sampled_window, start_ind = randomly_sample_window_from_signal(
                    first_sampled_portion, window_len
                )
            else:
                choice = randchoice([0, 1])
                if choice == 0:
                    sampled_window, start_ind = randomly_sample_window_from_signal(
                        first_sampled_portion, window_len
                    )
                elif choice == 1:
                    sampled_window, start_ind = randomly_sample_window_from_signal(
                        second_sampled_portion, window_len
                    )
                    start_ind = start_ind + tneg_end
                else:
                    raise ValueError("choice variable contains unhandled value;  choice=="+str(choice))

            return sampled_window, start_ind
    else:
        raise ValueError("Unrecognized window type requested for TS sample")
    pass

