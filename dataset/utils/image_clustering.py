import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ImageHistogramClustering:
    def __init__(self, root, max_samples, channels=[0, 1, 2], hist_bins=32, n_clusters=2):
        self.root = root
        self.max_samples = max_samples
        self.paths = os.listdir(root)[:max_samples]
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
        img = np.load(os.path.join(root, path)) * 255
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
        return self.model.labels_
    
    def print_element_clusters(self, ):
        print("Number of images in each cluster")
        for i in range(2):
            print(i, len([label for label in self.model.labels_ if label == i]))
    
    def show_cluster_images(self, cluster):
        paths = [path for path, label in zip(self.paths, self.model.labels_) if label == cluster][:10]
        fig, axs = plt.subplots(1, len(paths))
        for i, path in enumerate(paths):
            img = np.load(os.path.join(self.root, path))
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.show()

if __name__ == "__main__":
    root = "../melbourne/rgb_data/test"
    ic = ImageHistogramClustering(root, max_samples=1000, n_clusters=2)

    ic.cluster()
    ic.show_cluster_images(0)
    ic.show_cluster_images(1)
