import numpy as np
from time import time
import pandas as pd
from numba import jit

@jit(nopython=True)
def tanh(x):
    return np.tanh(x)

class Dense:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.in_size = input_size
        self.out_size = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.input @ self.weights + self.bias
        return self.output

    def create_another(self):
        return self.__class__((self.weights.shape), self.activation)

    def __str__(self):
        return f'{self.in_size} x {self.out_size}'

    __repr__ = __str__


class NeuralNetwork:
    def __init__(self, activation_func):
        self.layers = []
        self.columns_output = None
        self.activation_func = activation_func
        self.history_val = []
        self.history_train = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    def get_histories_of_costs():
        return self.history_train, self.history_val

    @staticmethod
    def convert_to_ndarray(data):
        """
        Converting pandas dataframe / series to numpy
        """
        columns = data.columns
        np_data = data.to_numpy()
        return np_data, columns

    @staticmethod
    def decode(y_pred):
        """
        Converting probabilities to binary
        """
        return [np.max(row) == row for row in y_pred]

    def train(self, X: pd.DataFrame or np.ndarray, y: pd.core.series.Series or np.ndarray, X_val: pd.DataFrame or np.ndarray, y_val: pd.core.series.Series or np.ndarray, generations, learning_rate, w, fi1, fi2, particles_in_swarm, verbose=False):
        """Train neural network.

        keyword arguments:
            X - predictors
            y - outcomes
            learning_rate - after learning_rate * generations without a change in global best cost stop training
            w - inertia parameter for particle swarm optimization
            fi1 - cognitive (local) acceleration
            fi2 - social (global) acceleration
            particles_in_swarm - number of particles in swarm
            verbose - printing param
        """
        assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray), 'Required input data format: pd.DataFrame'
        assert isinstance(y, pd.core.series.Series) or isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray), 'Required outcome data format: pd.DataFrame or pd.core.series.Series'
        assert len(X) == len(y), f'X length {len(X)} does not match y length {len(y)}'

        # converting to numpy
        if not isinstance(X, np.ndarray):
            X_np = X.to_numpy()
        else:
            X_np = X
        y_np, columns = NeuralNetwork.convert_to_ndarray(y)
        self.columns_output = columns

        if not isinstance(X_val, np.ndarray):
            X_val_np = X_val.to_numpy()
        else:
            X_val_np = X_val
        y_val_np, _ = NeuralNetwork.convert_to_ndarray(y_val)
        
        
        self.history_train, self.history_val = PCO(mse, particles_in_swarm, self).run(X_np, y_np, X_val_np, y_val_np, tanh, generations, learning_rate, w,  fi1, fi2, verbose)

    def predict(self, X):
        assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray), 'Required input data format: pd.DataFrame'
        
        if not isinstance(X, np.ndarray):
            X_np = X.to_numpy()
        else:
            X_np = X

        y_pred = np.zeros((X.shape[0], len(self.columns_output)))
        for i, sample in enumerate(X_np):
            result = sample
            for j, layer in enumerate(self.layers):
                result = layer.forward_propagation(result)
                if j != 0:
                    result = self.activation_func(result)

            y_pred[i] = result

        # Decode probabilities to binary, then convert to dataframe
        y_pred_pd = pd.DataFrame(NeuralNetwork.decode(y_pred), columns=self.columns_output)
        return y_pred_pd


@jit(nopython=True)
def mse(y_pred, y):
    return .5 * np.sum((y_pred - y)**2)


class Particle:
    def __init__(self, position, activation_func=tanh):
        self.position = position
        self.cost = float('inf')
        self.velocity = np.random.uniform(-1, 1, self.position.shape)

        self.lbest_pos = position
        self.lbest_cost = self.cost


        self.activation_func = activation_func


    # Useful for comparing particle objects
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        return self.cost <= other.cost


    def fold_vector(self, network, vector_len):
        """
        Weights and biases are stored in 1 dimensional vector. This method creates list of vectors of weights and biases for every layer
        """
        list_of_layers_properties = []

        j = 0
        for i in range(len(network.layers)):
            weights_and_biases = self.position[j:vector_len[i] + j]

            # separating layers
            layer_weight_bias = weights_and_biases.reshape((network.layers[i].in_size + 1, network.layers[i].out_size))

            # index of first bias
            treshold = network.layers[i].in_size

            # separating weights and biases
            weights = layer_weight_bias[:treshold]
            bias = layer_weight_bias[treshold:]

            list_of_layers_properties.append({'weights': weights, 'bias': bias})

            j += vector_len[i]

        return list_of_layers_properties

    def fold_vector_to_layer_final(self, network, vector_len):
        """
        copying global best position to the layers of given network
        """
        list_of_layer_props = self.fold_vector(network, vector_len)
        for i in range(len(network.layers)):
            network.layers[i].weights = list_of_layer_props[i]['weights']
            network.layers[i].bias = list_of_layer_props[i]['bias']


    def move(self, gbest_pos, w, fi1, fi2):
        """Move particle

        keyword arguments:
            gbest_post - global best position so fat
            w - inertia
            fi1 - cognitive (local) acceleration
            fi2 - social (global) acceleration
        """
        c1, c2 = np.random.rand(2, len(self.position))

        local_vel = fi1 * c1 * (self.lbest_pos - self.position)
        global_vel = fi2 * c2 * (gbest_pos - self.position)

        self.velocity = w * self.velocity + local_vel + global_vel

        self.position = self.position + self.velocity



    def __str__(self):
        return f'{self.cost}'

    __repr__ = __str__


class PCO:
    def __init__(self, cost_function, individuals, network):
        self.cost_function = cost_function
        self.vector_len = [layer.in_size * layer.out_size + layer.bias.shape[1] for layer in network.layers]
        self.network = network
        self.individuals = individuals
        self.swarm = self.init_swarm()
        self.gbest_pos = None
        self.gbest_particle = Particle(self.swarm[0].position, self.swarm[0].activation_func)
        self.gbest_cost = [float('inf')]

    def init_swarm(self):
        return np.array([Particle(position, self.network.activation_func) for position in np.random.uniform(-1, 1, (self.individuals, sum(self.vector_len)))])

    def get_cost(self, particle, X, y, network, vector_len):
        """Compute cost for current position of the particle

        keyword arguments:
            X - predictors
            y - outcomes
            network - blueprint of network that is being trained
            vector_len - list of indexes of layers
        """
        layers_form = particle.fold_vector(network, vector_len)
        cost = 0

        results = np.zeros(y.shape)
        for i in range(X.shape[0]):

            # forward propagation
            result = X[i]
            for j, layer in enumerate(layers_form):
                if j == 0:
                    result = result @ layer['weights'] + layer['bias']
                else:
                    result = particle.activation_func(result @ layer['weights'] + layer['bias'])
            max_idx = np.argmax(result)
            results[i, max_idx] = 1

            # add cost to overall network cost
            cost += mse(result, y[i]) / y.shape[0]

        acc = sum(np.all(results == y, axis=1)) / len(y)
        return cost, acc

    def update_particle_cost(self, particle, X, y, network, vector_len):
        cost, _ = self.get_cost(particle, X, y, network, vector_len)
        # update local best
        if cost < particle.lbest_cost:
            particle.lbest_pos = particle.position
            particle.lbest_cost = cost

        particle.cost = cost
        return cost

    def set_gbest_particle(self):
        """Set best global position if there is a new one"""
        particle = np.amin(self.swarm)

        if particle.cost < self.gbest_cost[-1]:
            self.gbest_pos = np.copy(particle.position)
            self.gbest_particle.position = np.copy(particle.position)
            self.gbest_cost.append(particle.cost)
            return True
        return False

    def run(self, X, y, val_X, val_y, activation_func, generations, learning_rate, w, fi1, fi2, verbose):
        val_cost_hist = []
        train_cost_hist = []
        # start timing
        t0 = time()
        print_generations_interval = int(generations * .25)
        stop_cond_index_behind = int(learning_rate * generations)

        # evaluate cost
        list(map(lambda particle: self.update_particle_cost(particle, X, y, self.network, self.vector_len), self.swarm))

        # best particle
        self.set_gbest_particle()
        gbest_particle_generation = 0

        for i in range(generations):

            for particle in self.swarm:
                # move particle
                particle.move(self.gbest_particle.position, w, fi1, fi2)

                # update cost of particle
                self.update_particle_cost(particle, X, y, self.network, self.vector_len)

#             self.swarm.sort()

            # settings global best particle
            if self.set_gbest_particle():
                gbest_particle_generation = i

            val_cost, val_acc = self.get_cost(self.gbest_particle, val_X, val_y, self.network, self.vector_len)
            train_cost, train_acc = self.get_cost(self.gbest_particle, X, y, self.network, self.vector_len)
            val_cost_hist.append(val_cost)
            train_cost_hist.append(train_cost)
#             print(train_cost_hist)

            if i - stop_cond_index_behind >= 0 and train_cost_hist[i - stop_cond_index_behind] - train_cost_hist[i] <= .001 or self.gbest_cost[-1] < .01:
                print(f'Generation {i} stop condition satisfied\nFinal cost: {self.gbest_cost[-1]}')
                break


            if verbose:
                print(f'Generation {i+1}/{generations}\tbest cost = {self.gbest_cost[-1]:.5f}\ttrain acc=>{train_acc:.3f}\tval acc=>{val_acc:.3f}')
            elif (i+1) % print_generations_interval == 0 or i == 0:
                print(f'Generation {i+1}/{generations}\tbest cost = {self.gbest_cost[-1]:.5f}')


        final_particle = Particle(self.gbest_pos)
        final_particle.fold_vector_to_layer_final(self.network, self.vector_len)

        # stop timing
        t1 = time()
        delta = t1 - t0

        print(f'\nTraining time: {delta:.2f}s')
        return self.gbest_cost, val_cost_hist
