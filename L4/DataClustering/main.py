import numpy as np
import matplotlib.pyplot as plt

def CNN(M, N, data):    # [M:number of clusters, N: number of data vectors]
    # [Initialize weights w1 and w2]
    # Find the centroid, c, of all data vector
    # Initialize w_1 and w_2 around c with small eps:
    c = np.mean(data, axis=0); eps = 0.1 * np.std(data, axis=0)
    W = np.array([c + eps, c - eps])
    Ns = np.zeros(2)
    k = 2; epoch = 0
    prev_winner = np.zeros(N, dtype=int)
    while k <= M:
        loser = 0
        for n in range(N):
            x = data[n] # Apply a data vector x(n) to the network
            j = np.argmin(np.sum((W - x) ** 2, axis=1)) # Find the winner neuron, j, in this epoch for 1<=j<=k
            Ns[j] += 1
            if epoch != 0:
                i = prev_winner[n] # Set i is Winner neuron, i, for x(n) in previous epoch.
                if i != j: # Then neuron i, is loser neuron
                    UpdateCNNWeight(x, W[i], W[j], Ns[i], Ns[j], epoch)
                    loser += 1
                    Ns[i] -= 1
            else:
                UpdateCNNWeight(x, None, W[j], None, Ns[j], epoch)
                loser += 1
            prev_winner[n] = j
        epoch += 1
        if loser == 0:
            break
        if k != M:
            # Split group with most error, j, by adding small vector e, nearby group j:
            j = np.argmin([sum([np.linalg.norm(xi - wi) ** 2 for xi in data]) for wi in W])
            W = np.vstack([W, W[j] + eps])
            Ns = np.append(Ns, 0)
        k += 1
    return W

def UpdateCNNWeight(x, Wi, Wj, Ni, Nj, epoch):
    # Update winner neuron : wj(n) := wj(n) + (1/Nj+1) * [x(n)- wj(n)]
    Wj += (1 / (Nj + 1)) * (x - Wj)
    if epoch != 0: # [loser neuron occurred only when epoch != 0]
        # Update loser neuron : wi(n) := wi(n) + (1/Ni-1) * [x(n)- wi(n)]
        Wi += (1 / (Wi - 1)) * (x - Wi)

def plot_CNN(data, W):
    # Assign each data point to its nearest centroid
    assignments = np.argmin(np.sum((data[:, np.newaxis, :] - W[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    # Plot the data points and color them according to their cluster assignments
    plt.scatter(data[:, 0], data[:, 1], c=assignments)
    # Plot the centroids
    plt.scatter(W[:, 0], W[:, 1], c='red', marker='x')
    plt.show()


# Load sample data
data = np.loadtxt('https://cs.joensuu.fi/sipu/datasets/s2.txt')
N = len(data)
# Set the number of clusters
M = 7
# Call the CNN function
W = CNN(M, N, data)
# Plot the results with colored clusters
plot_CNN(data, W)
