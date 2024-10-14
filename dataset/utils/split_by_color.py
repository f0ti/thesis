import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import cluster
from sklearn.cluster import KMeans

class ImageHistogramClustering:
    def __init__(self, dataset_name, max_samples=None, channels=[0, 1, 2], hist_bins=32, n_clusters=2):
        self.root = f"../{dataset_name}"
        self.rgb_dir = os.path.join(self.root, "rgb_data")
        self.max_samples = max_samples
        self.paths = os.listdir(self.rgb_dir)[:self.max_samples]
        self.n_clusters = n_clusters
        self.channels = channels
        self.hist_bins = hist_bins
        self.data = self.get_full_set_hist()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto")

    @staticmethod
    def show_rgb(img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def _get_image_hist(self, path, show=False):
        img = np.load(os.path.join(self.rgb_dir, path)) * 255
        # get averaged histogram of all channels
        hist = np.zeros((self.hist_bins, 3))
        for i in self.channels:
            hist[:, i], _ = np.histogram(img[:, :, i], bins=self.hist_bins, range=(0, 256))
        
        if show:
            plt.plot(hist)
            plt.axis('off')
            plt.show()

        return hist.flatten()

    def get_full_set_hist(self):
        return [self._get_image_hist(path) for path in self.paths]

    def cluster(self):
        X = np.array(self.data)
        self.model.fit(X)
        
        # save {path: label}
        self.clustered_paths = {path: label for path, label in zip(self.paths, self.model.labels_)}
        return self.clustered_paths
    
    def arrange_clustered_images(self):
        for i in range(self.n_clusters):
            print(f"Creating cluster_{i}")
            os.makedirs(f"{self.root}/cluster_{i}", exist_ok=True)
        
        for path, label in zip(self.paths, self.model.labels_):
            os.rename(os.path.join(self.root, "rgb_data", path), os.path.join(self.root, f"cluster_{label}", path))

        print("Images are arranged in clusters")

    # debugging functions

    def print_element_clusters(self, ):
        print("Number of images in each cluster")
        for i in range(self.n_clusters):
            print(i, len([label for label in self.model.labels_ if label == i]))
    
    def show_cluster_images(self, cluster):
        paths = [path for path, label in zip(self.paths, self.model.labels_) if label == cluster][:10]
        _, axs = plt.subplots(1, len(paths))
        for i, path in enumerate(paths):
            img = np.load(os.path.join(self.rgb_dir, path))
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.show()


class Splitter():
    def __init__(self, clusters, dataset_name="melbourne-top", data_type="xyz_data", n_clusters=2, split_ratio=0.8, del_artifacts=False):
        self.root = f"../{dataset_name}"
        self.split_ratio = split_ratio
        self.clusters = clusters
        self.data_type = f"{data_type}_data"
        self.n_clusters = n_clusters
        self.del_artifacts = del_artifacts

        assert cluster is not None, "Cluster is not defined"

    def split(self):
        # create train and test directories
        os.makedirs(os.path.join(self.root, "train", "rgb_data"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "train", self.data_type), exist_ok=True)
        os.makedirs(os.path.join(self.root, "test", "rgb_data"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "test", self.data_type), exist_ok=True)

        print("Splitting data...")
        for i in range(self.n_clusters):
            paths = [path for path, label in self.clusters.items() if label == i]
            np.random.shuffle(paths)
            split_idx = int(len(paths) * self.split_ratio)
            train_paths = paths[:split_idx]
            test_paths = paths[split_idx:]

            print(f"Cluster {i}: Train {len(train_paths)}, Test {len(test_paths)}")
            for path in train_paths:
                os.rename(os.path.join(self.root, "rgb_data", path), os.path.join(self.root, "train", "rgb_data", path))
                os.rename(os.path.join(self.root, self.data_type, path), os.path.join(self.root, "train", self.data_type, path))
            for path in test_paths:
                os.rename(os.path.join(self.root, "rgb_data", path), os.path.join(self.root, "test", "rgb_data", path))
                os.rename(os.path.join(self.root, self.data_type, path), os.path.join(self.root, "test", self.data_type, path))

        if self.del_artifacts:
            print("Deleting artifacts")
            os.rmdir(os.path.join(self.root, "rgb_data"))
            os.rmdir(os.path.join(self.root, self.data_type))

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    data_type = sys.argv[2]
    cluster_model = ImageHistogramClustering(dataset_name, n_clusters=2)
    clusters = cluster_model.cluster()

    splitter = Splitter(dataset_name=dataset_name, data_type=data_type, n_clusters=2, clusters=clusters, split_ratio=0.8, del_artifacts=True)
    splitter.split()
