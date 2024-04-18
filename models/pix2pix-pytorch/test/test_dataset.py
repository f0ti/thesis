import os
import unittest

# use unittest to test the functions

# check that in folder A there are only png images in train and test

class TestMakeDataset(unittest.TestCase):
    # get dataset paths
    def setUp(self):
        self.dataset_path = "../dataset/melbourne"

    def test_make_dataset(self):
        # check that in folder A there are only png images in train and test
        for split in ["train", "test"]:
            files = os.listdir(f"{self.dataset_path}/A/{split}")
            self.assertTrue(all(file.endswith(".png") for file in files))
            
        # check that in folder B there are only xyz files in train and test
        for split in ["train", "test"]:
            files = os.listdir(f"{self.dataset_path}/B/{split}")
            self.assertTrue(all(file.endswith(".npz") for file in files))
        
        # check that the number of files in A train is equal to the number of files in B train
        files_A_train = os.listdir(f"{self.dataset_path}/A/train")
        files_B_train = os.listdir(f"{self.dataset_path}/B/train")
        self.assertEqual(len(files_A_train), len(files_B_train))
        
        # check that the number of files in A test is equal to the number of files in B test
        files_A_test = os.listdir(f"{self.dataset_path}/A/test")
        files_B_test = os.listdir(f"{self.dataset_path}/B/test")
        self.assertEqual(len(files_A_test), len(files_B_test))

unittest.main(argv=[''], verbosity=2, exit=False)
