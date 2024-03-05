Created by: Foti
Created time: February 21, 2024 9:54 AM
Tags: 3D Projection, Indoor Scenes, Outdoor Scenes
## Approach
This paper uses post-scan colorization based on later collected images of the scanned site.

It presents a simple but efficient point cloud colorization method based on a point-to-pixel orthogonal projection under an assumption that the orthogonal and perspective projections can produce similar effects for a planar feature as long as the target-to-camera distance is relatively short (within several meters).
## Results
This assumption was verified by a simulation experiment, and the results show that only approximately 5% of colorization error was found at a target-to-camera distance of 3 m. The method was further verified with two real datasets collected for the cultural heritage documentation. The results showed that the visuality of the point clouds for two giant historical buildings had been greatly improved after applying the proposed method.