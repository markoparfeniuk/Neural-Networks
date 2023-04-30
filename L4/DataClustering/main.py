import numpy as np
import matplotlib.pyplot as plt

def cent_nn(X, k):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k)]
    while True:
        # Assign each data point to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # Check if centroids have converged
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels

data = np.loadtxt("http://cs.joensuu.fi/sipu/datasets/s2.txt")
labels = cent_nn(data, 6)
print(labels);

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()