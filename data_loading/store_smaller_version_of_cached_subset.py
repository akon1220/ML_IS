import os
import argparse
import pickle as pkl
import random

from utils.caching_utils import create_directory

def get_sample(sample_loc_tuple):
    curr_sample = None
    with open(sample_loc_tuple[0], "rb") as infile:
        temp_sample_set = pkl.load(infile)
        curr_sample = temp_sample_set[sample_loc_tuple[1]]
    return curr_sample

def cache_smaller_subset(subset_path, reduced_subset_save_dir, max_data_samp_ratio=None, random_seed=0):
    print("cache_smaller_subset: Original subset path == ", subset_path)

    curr_subset_sample_index_map = {} # self.curr_subset_sample_index_map = {}
    curr_index_counter = 0
    cached_sample_file_names = os.listdir(subset_path)
    max_num_samples_in_file = 0
    for file_name in cached_sample_file_names:
        with open(subset_path + os.sep + file_name, "rb") as infile:
            curr_samples = pkl.load(
                infile
            )  # this is assumed to be a list of length num_samples_in_file
            num_samples_in_file = len(curr_samples)
            if num_samples_in_file > max_num_samples_in_file:
                max_num_samples_in_file = num_samples_in_file
            for j in range(num_samples_in_file):
                curr_subset_sample_index_map[curr_index_counter] = (
                    subset_path + os.sep + file_name,
                    j,
                )
                # self.curr_subset_sample_index_map[curr_index_counter] = (
                #     subset_path + os.sep + file_name,
                #     j,
                # )
                curr_index_counter += 1
                pass
            pass
        pass

    print("cache_smaller_subset: Original number of samples in subset == ", len(curr_subset_sample_index_map.keys()))
    if max_data_samp_ratio is not None:
        reduced_subset_sample_index_map = {}
        curr_sample_indices = list(curr_subset_sample_index_map.keys())
        assert len(curr_sample_indices) == len(curr_subset_sample_index_map.keys()) # sanity check
        num_samples_to_keep = int(len(curr_sample_indices)*max_data_samp_ratio)
        random.Random(random_seed).shuffle(curr_sample_indices) # see https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
        for new_ind, old_ind in enumerate(curr_sample_indices[:num_samples_to_keep]):
            reduced_subset_sample_index_map[new_ind] = curr_subset_sample_index_map[old_ind]
        curr_subset_sample_index_map = reduced_subset_sample_index_map
        pass

    num_samples_in_curr_subset = len(curr_subset_sample_index_map.keys())
    print("cache_smaller_subset: Final number of samples in subset == ", num_samples_in_curr_subset)

    curr_subset_id_counter = 0
    curr_subset = []
    curr_subset_save_path = os.sep.join([reduced_subset_save_dir, "subset" + str(curr_subset_id_counter) + ".pkl"])
    for sample_to_keep_id in list(curr_subset_sample_index_map.keys()):
        # draw sample
        curr_subset.append(get_sample(curr_subset_sample_index_map[sample_to_keep_id]))

        # check if current subset needs to be cached
        if len(curr_subset) == max_num_samples_in_file:
            with open(curr_subset_save_path, "wb") as outfile:
                pkl.dump(curr_subset, outfile)
            # initialize new subset
            curr_subset_id_counter += 1
            curr_subset = []
            curr_subset_save_path = os.sep.join([reduced_subset_save_dir, "subset" + str(curr_subset_id_counter) + ".pkl"])

    return curr_subset_sample_index_map

if __name__ == "__main__":
    """
    preprocess_TUHAbnormal_data.__main__: shuffles, splits, and caches TUAB data based on provided arguments
    - Inputs:
       * cached_args_file (argparse obj pickle file): arguments for preprocessing the original TUAB dataset, including
          - cached_args.root_save_directory (str): the directory containing the original cached dataset
          - cached_args.max_data_samp_ratios_by_split_type (list(float)): a list of ratios representing how many data samples to keep from each split, formated as [train_ratio, val_ratio, test_ratio]  
          - cached_args.new_save_directory (str): the directory to save the new dataset to
    - Outputs:
       * */cached_samples*.pkl (cached list): pickle files containing lists of cached samples, with each sample formatted as (x1, x2, ..., xn, label_y)
    - Usage:
       * PreprocessedDataset.__init__: uses TUHAbnormalDataset.preprocess_and_cache_data when self.data_save_directory is empty
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cached_args_file",
        type=str,
        default="cached_args_resample_data.pkl",
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

    create_directory(cached_args.new_save_directory)

    # see https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    files_to_copy = [f for f in os.listdir(cached_args.root_save_directory) if os.path.isfile(os.path.join(cached_args.root_save_directory, f))]
    splits_to_subsample = [p for p in os.listdir(cached_args.root_save_directory) if os.path.isdir(os.path.join(cached_args.root_save_directory, p))]

    for filename in files_to_copy:
        orig_contents = None
        with open(os.path.join(cached_args.root_save_directory, filename), 'rb') as infile:
            orig_contents = pkl.load(infile)
        with open(os.path.join(cached_args.new_save_directory, "orig_"+filename), 'wb') as outfile:
            pkl.dump(orig_contents, outfile)
    
    for split_dir_name in splits_to_subsample:
        old_split_subset_path = os.path.join(cached_args.root_save_directory, split_dir_name)
        new_split_subset_path = os.path.join(cached_args.new_save_directory, split_dir_name)
        create_directory(new_split_subset_path)
        curr_data_samp_ratio = None
        if "train" in old_split_subset_path:
            curr_data_samp_ratio = cached_args.max_data_samp_ratios_by_split_type[0]
        elif "validation" in old_split_subset_path:
            curr_data_samp_ratio = cached_args.max_data_samp_ratios_by_split_type[1]
        elif "test" in old_split_subset_path:
            curr_data_samp_ratio = cached_args.max_data_samp_ratios_by_split_type[2]
        else:
            raise ValueError("Unsupported old_split_subset_path == "+str(old_split_subset_path))
        subsample_map = cache_smaller_subset(old_split_subset_path, new_split_subset_path, max_data_samp_ratio=curr_data_samp_ratio, random_seed=0)
        with open(os.path.join(cached_args.new_save_directory, split_dir_name+"_subsample_map.pkl"), 'wb') as outfile:
            pkl.dump(subsample_map, outfile)

    with open("subsampling_cached_args.pkl", "wb") as outfile:
        pkl.dump(cached_args, outfile)

    print("<<< MAIN: DONE RUNNING !!!")
    pass