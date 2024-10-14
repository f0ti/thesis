import os
import fire
import test

class Splitter():
    def __init__(
        self,
        dataset_name = "estonia-i",
        data_type = "i_data",
        split_ratio = 0.8,
        del_artifacts = False
    ):
        self.root = f"../{dataset_name}"
        self.data_type = data_type
        self.rgb_data_path = os.path.join(self.root, "rgb_data")
        self.coo_data_path = os.path.join(self.root, self.data_type)

        self.del_artifacts = del_artifacts
        self.split_ratio = split_ratio

        self.rgb_data = sorted(os.listdir(self.rgb_data_path))
        self.coo_data = sorted(os.listdir(self.coo_data_path))
        
        print(len(self.rgb_data), len(self.coo_data))

    def group_data(self):
        rgb_identifiers_name = set()
        self.rgb_identifiers = {} # keeps counter of the number of files for each identifier
        for file in self.rgb_data:
            rgb_identifiers_name.add(file.split("_")[0])
            identifier = file.split("_")[0]
            if identifier in self.rgb_identifiers:
                self.rgb_identifiers[identifier].append(file)
            else:
                self.rgb_identifiers[identifier] = [file]

        coo_identifiers_name = set()
        self.coo_identifiers = {} # keeps counter of the number of files for each identifier
        for file in self.coo_data:
            coo_identifiers_name.add(file.split("_")[0])
            identifier = file.split("_")[0]
            if identifier in self.coo_identifiers:
                self.coo_identifiers[identifier].append(file)
            else:
                self.coo_identifiers[identifier] = [file]
        
        assert len(rgb_identifiers_name) == len(coo_identifiers_name), "Number of identifiers do not match"

    def delete_artifacts(self):
        print("Deleting artifacts")
        os.rmdir(os.path.join(self.root, "rgb_data"))
        os.rmdir(os.path.join(self.root, self.data_type))

    def split(self):
        os.makedirs(os.path.join(self.root, "train", "rgb_data"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "train", self.data_type), exist_ok=True)
        os.makedirs(os.path.join(self.root, "test", "rgb_data"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "test", self.data_type), exist_ok=True)

        num_identifiers = 190
        print(f"Number of identifiers: {num_identifiers}")

        self.group_data()

        # calculate the threshold for the split as a factor of the number of files = 190
        threshold = int(len(self.rgb_data) * self.split_ratio) // 190
        print(f"Threshold: {threshold}")
        
        # pass the first threshold number of files to the train set and the rest to the test set
        train_identifiers = list(self.rgb_identifiers)[:threshold]
        test_identifiers = list(self.rgb_identifiers)[threshold:]

        for identifier in test_identifiers:
            print(f"Processing identifier {identifier} in test set")
            for file in self.rgb_identifiers[identifier]:
                os.rename(os.path.join(self.rgb_data_path, file), os.path.join(self.root, "test", "rgb_data", file))
                os.rename(os.path.join(self.coo_data_path, file), os.path.join(self.root, "test", self.data_type, file))

        for identifier in train_identifiers:
            print(f"Processing identifier {identifier} in train set")
            for file in self.rgb_identifiers[identifier]:
                os.rename(os.path.join(self.rgb_data_path, file), os.path.join(self.root, "train", "rgb_data", file))
                os.rename(os.path.join(self.coo_data_path, file), os.path.join(self.root, "train", self.data_type, file))

        if self.del_artifacts:
            self.delete_artifacts()

if __name__ == "__main__":
    fire.Fire(Splitter)