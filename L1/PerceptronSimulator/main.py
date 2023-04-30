import numpy as np

class Perceptron:
    def __init__(self):
        # Ініціалізація ваг та зсувів з випадковими значеннями
        self.weights1 = np.random.uniform(-3, 3, (2, 2))
        self.weights2 = np.random.uniform(-3, 3, (2, 1))
        self.bias1 = np.random.uniform(-3, 3, (1, 2))
        self.bias2 = np.random.uniform(-3, 3, (1, 1))

    def sigmoid(self, x):
        # Функція активації
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Функція прямого ходу
        hidden_layer = self.sigmoid(np.dot(x, self.weights1) + self.bias1)
        output_layer = self.sigmoid(np.dot(hidden_layer, self.weights2) + self.bias2)
        return hidden_layer, output_layer

    def predict(self, x):
        # Функція передбачення на основі прямого ходу
        output_layer = self.forward(x)
        return np.round(output_layer)

    def train(self, x, y, epochs=1000, learning_rate=0.1):
        # Навчання з використанням backpropagation
        for epoch in range(epochs):
            hidden_layer, output_layer = self.forward(x)
            error = y - output_layer # Обчислення помилки на виході
            d_output = error * output_layer * (1 - output_layer)  # Похідна вихідного шару
            d_hidden = np.dot(d_output, self.weights2.T) * hidden_layer * (1 - hidden_layer) # Похідна прихованого шару
            # Оновлюємо ваги та зсуви
            self.weights2 += learning_rate * np.dot(hidden_layer.T, d_output)
            self.weights1 += learning_rate * np.dot(x.T, d_hidden)
            self.bias2 += learning_rate * np.dot(np.ones((1, x.shape[0])), d_output)
            self.bias1 += learning_rate * np.dot(np.ones((1, x.shape[0])), d_hidden)


x = np.array([[3, -3], [-3, -3], [3, 3], [-3, 3]])
y = np.array([[0], [1], [1], [0]])

perceptron = Perceptron()
perceptron.train(x, y)
predictions = perceptron.predict(x)
print(predictions)
