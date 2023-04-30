import numpy as np
import matplotlib.pyplot as plt

class HammingNetwork:
    def __init__(self, reference_vectors):
        self.reference_vectors = reference_vectors

    def calculate_hamming_distance(self, input_vector):
        hamming_distances = []
        for reference_vector in self.reference_vectors:
            difference = reference_vector != input_vector
            hamming_distance = np.sum(difference)
            hamming_distances.append(hamming_distance)
        return np.array(hamming_distances)

    def classify(self, input_vector):
        hamming_distances = self.calculate_hamming_distance(input_vector)
        return np.argmin(hamming_distances)

    def similarity(self, input_vector):
        hamming_distances = self.calculate_hamming_distance(input_vector)
        similarities = 1 - hamming_distances / len(input_vector)
        return similarities

    def plot_vector(self, vector):
        matrix = vector.reshape((4, 4))
        plt.imshow(matrix, cmap='gray_r')
        plt.show()

# Example usage
reference_vectors = [
    np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
    np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1])
]
hamming_network = HammingNetwork(reference_vectors)

input_vector = np.random.randint(0, 1 + 1, 16)

hamming_network.plot_vector(reference_vectors[0])
hamming_network.plot_vector(reference_vectors[1])
hamming_network.plot_vector(input_vector)

print(f"ref 0 vector: {reference_vectors[0]}")
print(f"ref 1 vector: {reference_vectors[1]}")
print(f"input vector: {input_vector}")
classification = hamming_network.classify(input_vector)
print(f"\nclassification: reference #{classification}")
similarity0, similarity1 = HammingNetwork.similarity(hamming_network, input_vector)
print(f"\nreference/similarity\n - ref[0]: {similarity0 * 100}%\n - ref[1]: {similarity1 * 100}%");
