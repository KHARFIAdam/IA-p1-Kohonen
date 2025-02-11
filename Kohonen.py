import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """ Classe représentant un neurone dans une carte de Kohonen """
    
    def __init__(self, w, posx, posy):
        self.weights = w.flatten()
        self.posx = posx
        self.posy = posy
        self.y = 0.
    
    def compute(self, x):
        """ Calcule la distance entre l'entrée et les poids du neurone """
        self.y = np.linalg.norm(self.weights - x)
    
    def learn(self, eta, sigma, posxbmu, posybmu, x):
        """ Met à jour les poids du neurone en fonction de la règle de Kohonen """
        distance_to_bmu = (self.posx - posxbmu)**2 + (self.posy - posybmu)**2
        influence = np.exp(-distance_to_bmu / (2 * sigma**2))
        self.weights += eta * influence * (x - self.weights)

class KohonenSOM:
    """ Carte auto-organisatrice de Kohonen """
    
    def __init__(self, input_dim=2, grid_size=(10, 10), lr=0.1, sigma=1.0, n_iterations=1000):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = lr
        self.sigma = sigma
        self.n_iterations = n_iterations
        self.map = [[Neuron(np.random.rand(input_dim), i, j) for j in range(grid_size[1])] for i in range(grid_size[0])]
    
    def find_winner(self, sample):
        """ Trouve le neurone dont les poids sont les plus proches de l'entrée """
        bmu = None
        min_dist = float('inf')
        for row in self.map:
            for neuron in row:
                neuron.compute(sample)
                if neuron.y < min_dist:
                    min_dist = neuron.y
                    bmu = neuron
        return bmu.posx, bmu.posy
    
    def train(self, data):
        """ Entraîne la carte de Kohonen """
        for i in range(self.n_iterations):
            sample = data[np.random.randint(0, len(data))]
            bmu_x, bmu_y = self.find_winner(sample)
            decay_factor = np.exp(-i / self.n_iterations)
            lr_t = self.lr * decay_factor
            sigma_t = self.sigma * decay_factor
            for row in self.map:
                for neuron in row:
                    neuron.learn(lr_t, sigma_t, bmu_x, bmu_y, sample)
    
    def plot_weights(self):
        """ Visualisation des poids de la carte après entraînement """
        fig, ax = plt.subplots()
        for row in self.map:
            for neuron in row:
                ax.scatter(neuron.weights[0], neuron.weights[1], color='red')
        ax.set_title("Carte de Kohonen après apprentissage")
        plt.show()

# Génération de données d'apprentissage
data = np.random.uniform(-1, 1, (500, 2))

# Initialisation et entraînement du réseau SOM
som = KohonenSOM()
som.train(data)
som.plot_weights()
