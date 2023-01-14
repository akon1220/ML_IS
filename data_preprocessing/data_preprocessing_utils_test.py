# python_code_templates.py: a file for testing/debugging data_preprocessing_utils.py
# SEE LICENSE STATEMENT AT THE END OF THE FILE

# dependency import statements
import unittest
import os
import shutil
import numpy as np
from data_preprocessing_utils import *
from utils.caching_utils import create_directory

curr_dir_path = os.path.dirname(
    os.path.realpath(__file__)
)  # https://stackoverflow.com/questions/5137497/find-the-current-directory-and-files-directory


class TestPreprocessedDataset(unittest.TestCase):
    """
    TestPreprocessedDataset Description:
    - General Purpose: this class tests the PreprocessedDataset member functions (esp. __init__)
    - Usage:
      * terminal: run __main__ from terminal to perform debugging tests
    """

    def setUp(self):
        """
        TestPreprocessedDataset.setUp: creates a mock PreprocessedDataset object using fake datasets
         - Inputs:
            * N/A
         - Outputs:
            * N/A
         - Usage:
            * data_preprocessing_utils_test.__main__(): calls TestPreprocessedDataset.setUp at beginning of tests
        """
        self.root_save_dir1 = curr_dir_path + os.sep + "temp_test_dataset1"
        self.root_save_dir2 = curr_dir_path + os.sep + "temp_test_dataset2"
        data_split_type = "crossValSplit"
        task_id = "RP"
        data_source_id = "CPNEFluoxetine"
        date_of_data_split = "YYYYMMDD"

        # test __init__ of new dataset
        self.preprocessedDataset1 = PreprocessedDataset(
            self.root_save_dir1,
            data_split_type,
            task_id,
            data_source_id,
            date_of_data_split,
            data_source_specific_args=None,
            num_splits=1,
            holdout_set=False,
            split_ratios=(0.7, 0.2, 0.1),
        )

        # test __init__ of existing preprocessed dataset
        create_directory(self.root_save_dir2)
        root_save_dir2_train1_set = self.root_save_dir2 + os.sep + "train1"
        create_directory(root_save_dir2_train1_set)
        root_save_dir2_val1_set = self.root_save_dir2 + os.sep + "validation1"
        create_directory(root_save_dir2_val1_set)
        root_save_dir2_test1_set = self.root_save_dir2 + os.sep + "test1"
        create_directory(root_save_dir2_test1_set)

        train1_batchset1 = [
            [
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 2, (1)),
            ]
            for _ in range(10)
        ]  # see https://stackoverflow.com/questions/19984596/numpy-array-of-random-matrices
        train1_batchset2 = [
            [
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 2, (1)),
            ]
            for _ in range(5)
        ]
        val1_batchset1 = [
            [
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 2, (1)),
            ]
            for _ in range(10)
        ]
        val1_batchset2 = [
            [
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 2, (1)),
            ]
            for _ in range(2)
        ]
        test1_batchset1 = [
            [
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 2, (1)),
            ]
            for _ in range(10)
        ]
        test1_batchset2 = [
            [
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 100, (1, 25, 100)),
                np.random.randint(0, 2, (1)),
            ]
            for _ in range(8)
        ]

        with open(root_save_dir2_train1_set + os.sep + "train0.pkl", "wb") as outfile:
            pkl.dump(train1_batchset1, outfile)
        with open(root_save_dir2_train1_set + os.sep + "train1.pkl", "wb") as outfile:
            pkl.dump(train1_batchset2, outfile)
        with open(
            root_save_dir2_val1_set + os.sep + "validation0.pkl", "wb"
        ) as outfile:
            pkl.dump(val1_batchset1, outfile)
        with open(
            root_save_dir2_val1_set + os.sep + "validation1.pkl", "wb"
        ) as outfile:
            pkl.dump(val1_batchset2, outfile)
        with open(root_save_dir2_test1_set + os.sep + "test0.pkl", "wb") as outfile:
            pkl.dump(test1_batchset1, outfile)
        with open(root_save_dir2_test1_set + os.sep + "test1.pkl", "wb") as outfile:
            pkl.dump(test1_batchset2, outfile)

        self.preprocessedDataset2 = PreprocessedDataset(
            self.root_save_dir2,
            data_split_type,
            task_id,
            data_source_id,
            date_of_data_split,
            data_source_specific_args=None,
            num_splits=1,
            holdout_set=False,
            split_ratios=(0.7, 0.2, 0.1),
        )
        pass

    def tearDown(self):
        """
        TestPreprocessedDataset.tearDown: un-does any changes induced by TestPreprocessedDataset tests
         - Inputs:
            * N/A
         - Outputs:
            * N/A
         - Usage:
            * data_preprocessing_utils_test.__main__(): calls TestPreprocessedDataset.tearDown at end of tests
        """
        self.preprocessedDataset1.dispose()
        self.preprocessedDataset2.dispose()
        shutil.rmtree(
            self.root_save_dir1
        )  # https://stackoverflow.com/questions/13118029/deleting-folders-in-python-recursively
        shutil.rmtree(self.root_save_dir2)
        pass


if __name__ == "__main__":
    """
    data_preprocessing_utils_test.__main__: runs various tests on TestPreprocessedDataset class definition
     - Inputs:
        * N/A
     - Outputs:
        * N/A
     - Usage:
        * called from command line
    """
    unittest.main()
    pass

