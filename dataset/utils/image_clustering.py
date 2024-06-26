import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

    def get_image_hist(self, path, show=False):
        img = np.load(os.path.join(self.rgb_dir, path)) * 255
        for i in self.channels:
            hist = np.histogram(img[i], bins=self.hist_bins, range=(0, 256))[0]
            if show:
                plt.plot(hist)
                plt.axis('off')
                plt.show()
        return hist

    def get_full_set_hist(self):
        return [self.get_image_hist(path) for path in self.paths]

    def cluster(self):
        X = np.array(self.data)
        self.model.fit(X)
        # save path and label as dict
        self.clustered_paths = {path: label for path, label in zip(self.paths, self.model.labels_)}
        return self.clustered_paths
    
    def print_element_clusters(self, ):
        print("Number of images in each cluster")
        for i in range(self.n_clusters):
            print(i, len([label for label in self.model.labels_ if label == i]))
    
    def show_cluster_images(self, cluster):
        paths = [path for path, label in zip(self.paths, self.model.labels_) if label == cluster][:10]
        fig, axs = plt.subplots(1, len(paths))
        for i, path in enumerate(paths):
            img = np.load(os.path.join(self.rgb_dir, path))
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.show()

    def arrange_clustered_images(self):
        for i in range(self.n_clusters):
            print(f"Creating cluster_{i}")
            os.makedirs(f"{self.root}/cluster_{i}", exist_ok=True)
        
        for path, label in zip(self.paths, self.model.labels_):
            os.rename(os.path.join(self.root, "rgb_data", path), os.path.join(self.root, f"cluster_{label}", path))


class Splitter():
    def __init__(self, max_samples=None, dataset_name="melbourne-top", n_clusters=2, split_ratio=0.8, del_artifacts=False):
        self.ihc = ImageHistogramClustering(dataset_name, max_samples, n_clusters=n_clusters)
        self.ihc.cluster()
        self.split_ratio = split_ratio
        self.del_artifacts = del_artifacts

    def split(self):
        # create train and test directories
        os.makedirs(os.path.join(self.ihc.root, "train", "rgb_data"), exist_ok=True)
        os.makedirs(os.path.join(self.ihc.root, "train", "xyz_data"), exist_ok=True)
        os.makedirs(os.path.join(self.ihc.root, "test", "rgb_data"), exist_ok=True)
        os.makedirs(os.path.join(self.ihc.root, "test", "xyz_data"), exist_ok=True)

        for i in range(self.ihc.n_clusters):
            paths = [path for path, label in self.ihc.clustered_paths.items() if label == i]
            np.random.shuffle(paths)
            split_idx = int(len(paths) * self.split_ratio)
            train_paths = paths[:split_idx]
            test_paths = paths[split_idx:]

            print(f"Cluster {i}: Train {len(train_paths)}, Test {len(test_paths)}")
            for path in train_paths:
                os.rename(os.path.join(self.ihc.root, "rgb_data", path), os.path.join(self.ihc.root, "train", "rgb_data", path))
                os.rename(os.path.join(self.ihc.root, "xyz_data", path), os.path.join(self.ihc.root, "train", "xyz_data", path))
            for path in test_paths:
                os.rename(os.path.join(self.ihc.root, "rgb_data", path), os.path.join(self.ihc.root, "test", "rgb_data", path))
                os.rename(os.path.join(self.ihc.root, "xyz_data", path), os.path.join(self.ihc.root, "test", "xyz_data", path))

        if self.del_artifacts:
            print("Deleting artifacts")
            os.rmdir(os.path.join(self.ihc.root, "rgb_data"))
            os.rmdir(os.path.join(self.ihc.root, "xyz_data"))

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    cluster = ImageHistogramClustering(dataset_name, n_clusters=2)
    cluster.cluster()

    splitter = Splitter(dataset_name=dataset_name, n_clusters=2, split_ratio=0.8, del_artifacts=True)
    splitter.split()
