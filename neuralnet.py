import numpy as np
from time import time
import pandas as pd
from collections.abc import Callable
from typing import List, Dict, Tuple


class Dense:
    # input_size = number of neurons in previous layer or features in sample
    # output_size = number of neurons in this layer
    def __init__(self, input_size: int, output_size: int, activation_func: Callable[np.ndarray, np.ndarray]):
        self.in_size = input_size
        self.out_size = output_size
        self.weights = None
        self.bias = None
        self.activation_func = activation_func

    # returns output for a given input
    def linear_forward(self, A: np.ndarray, w: np.ndarray=None, b: np.ndarray=None) -> np.ndarray:
        """Implementation of the linear part of a layer's forward propagation."""        
        
        if w is None and b is None:
            w = self.weights
            b = self.bias
            
        return A @ w + b
    
    def activation_linear_forward(self, A: np.ndarray, w: np.ndarray=None, b: np.ndarray=None) -> np.ndarray:
        """Implementation of froward propagation for the Linear->Activation layer"""
        return self.activation_func(self.linear_forward(A, w, b))

    def __str__(self):
        return f'{self.in_size} x {self.out_size}'
    
    def init_parameters(self, init_how: str) -> np.ndarray:
        """Initializing weights and bias parameters."""
        xavier_he = {'xavier': np.sqrt(1/self.in_size), 'he_normal': np.sqrt(2/self.in_size)}
        
        w = np.random.randn(self.in_size, self.out_size) * xavier_he[init_how]
        b = np.random.rand(1, self.in_size) * 0.2
        
        flat_w = w.reshape(1, self.in_size * self.out_size)
        
        return np.concatenate((flat_w, b), axis=1)
    
        
    
    __repr__ = __str__


class NeuralNetwork:
    def __init__(self, cost_func: Callable[np.ndarray, np.ndarray], how_init_weights: str, caption: str="Neural network"):
        self.layers = []
        self.columns_output = None
        self.cost_func = cost_func
        self.how_init_weights = how_init_weights
        self.history_val = []
        self.history_train = []
        self.caption = caption

    # add layer to network
    def next_layer(self, layer):
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
    
    def init_parameters(self) -> np.ndarray:
        """
        Method which initialize parameters w and b for each layer in network.
        
        Return
        ------
            np.ndarray, shape [1 x n_parameters_in_the_neural_network]
            Flattened numpy array with parameters for the whole network 
        """
        params = self.layers[0].init_parameters(self.how_init_weights)
        for layer in self.layers[1:]:
            params = np.concatenate((params, layer.init_parameters(self.how_init_weights)), axis=None)
        
        return params
        
    def forward_propagation(self, X: np.ndarray, parameters: np.ndarray=None) -> np.ndarray:
        """
        Implementation of forward propagation 
        
        Arguments
        ---------
            X - dataset
            parameters - flattened numpy array with parameters for the whole network 
                If the argument was not passed it means forward propagation will be computed with trained parameters.
        
        Return
        ------
            np.ndarray, shape [n_samples_in_X x last_layer_out_size]
        """

        result = X
        if parameters is None:
            for j, layer in enumerate(self.layers):
                result = layer.activation_linear_forward(result)
        else: 
            for j, layer in enumerate(self.layers):
                result = layer.activation_linear_forward(result, w=parameters[j]['weights'], b=parameters[j]['bias'])
            
        return result
        

    def train(self, X: pd.DataFrame or np.ndarray, y: pd.core.series.Series or np.ndarray, X_val: pd.DataFrame or np.ndarray, y_val: pd.core.series.Series or np.ndarray, generations, w, fi1, fi2, particles_in_swarm, patience: int=None, min_delta: float=None, verbose=False):
        """
        Training neural network with pso optimization algorithm.

        Parameters:
            X - training set with predictiors
            y - training set with outcome
            val_X - validation set with predictions
            val_y - validation set with outcome
            generations - how many iterations of pso will be executed
            patience - generations without significant improvement, after that learning loop is stopped 
            min_delta - significant improvement between i-th generation cost and (i - patience)-th generation cost
            w - hyperparameter inertia
            fi1 - hyperparameter cognitive(local) acceleration
            fi2 - hyperparameter social(global) acceleration
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
        
        
        self.history_train, self.history_val = PSO(particles_in_swarm, self).run(X_np, y_np, X_val_np, y_val_np, generations, w,  fi1, fi2, patience, min_delta, verbose)

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predicts outcome for given X (basically runs forward propagation)
        
        X - np.ndarray or pandas dataframe which contains n observations
        
        Returns:
            y_pred_pd - pandas Dataframe with predictions
        """
        assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray), 'Required input data format: pd.DataFrame'
        
        if not isinstance(X, np.ndarray):
            X_np = X.to_numpy()
        else:
            X_np = X

        y_pred = self.forward_propagation(X_np)

        # Decode probabilities to binary, then convert to dataframe
        y_pred_pd = pd.DataFrame(NeuralNetwork.decode(y_pred), columns=self.columns_output)
        return y_pred_pd


def mse(y_pred, y):
    return np.sum((y_pred - y)**2) / len(y)

def cross_entropy(y_pred, y):
    return -np.sum(y * np.log(y_pred))

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    return  np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
def relu(x):
    return x * (x > 0)
    
class Particle:
    def __init__(self, position):
        """
        Particle constructor
        
        Parameters:
            position - in other words that is just a vector of weights and biases
        """
        self.position = position
        self.cost = float('inf')
        self.velocity = np.random.uniform(-1, 1, self.position.shape)

        self.lbest_pos = position
        self.lbest_cost = self.cost


    # Useful for comparing particle objects
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        return self.cost <= other.cost


    def fold_vector(self, network: NeuralNetwork, vector_len: List[int]) -> List[Dict[str, np.ndarray]]:
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

    def fold_vector_to_layer_final(self, network: NeuralNetwork, vector_len: List[int]):
        """
        Copying global best position to the layers of given network
        """
        list_of_layer_props = self.fold_vector(network, vector_len)
        for i in range(len(network.layers)):
            network.layers[i].weights = list_of_layer_props[i]['weights']
            network.layers[i].bias = list_of_layer_props[i]['bias']


    def move(self, gbest_pos: np.ndarray, w: float, fi1: float, fi2: float):
        """Move particle

         Parameters:
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


class PSO:
    def __init__(self, individuals: int, network: NeuralNetwork):
        """
        Particle swarm optimization class
        
        Parameters:
            individuals - number of particles in swarm
            network - neural network object
        """
        
        self.vector_len = [layer.in_size * layer.out_size + layer.out_size for layer in network.layers]
        self.network = network
        self.individuals = individuals
        self.swarm = self.init_swarm(network)
        self.gbest_pos = None
        self.gbest_particle = Particle(self.swarm[0].position)
        self.gbest_cost = [float('inf')]

    def init_swarm(self, network: NeuralNetwork):
        """
        Initialization of swarm with random weights
        """
        
        return np.array([Particle(network.init_parameters()) for _ in range(self.individuals)])

    def get_cost(self, particle, X, y, network, vector_len):
        """Compute cost for current position of the particle

        Parameters:
            particle - single particle (vector of weights and biases of the neural network)
            X - predictors
            y - outcomes
            network - architecture of network that is being trained
            vector_len - list of indexes of layers
            
        Returns:
            * cost
            * accuracy 
        """
        parameters = particle.fold_vector(network, vector_len)
        
        AL = network.forward_propagation(X, parameters)

        # add loss to overall network cost
        cost = network.cost_func(AL, y)
        return cost, AL

    def update_particle_cost(self, particle, X, y, network, vector_len):
        cost, _ = self.get_cost(particle, X, y, network, vector_len)
        # update local best
        if cost < particle.lbest_cost:
            particle.lbest_pos = particle.position
            particle.lbest_cost = cost

        particle.cost = cost

    def set_gbest_particle(self):
        """Set best global position if there is a new one"""
        particle = min(self.swarm)

        if particle.cost < self.gbest_cost[-1]:
            self.gbest_pos = np.copy(particle.position)
            self.gbest_particle.position = np.copy(particle.position)
            self.gbest_cost.append(particle.cost)
            return True
        return False
    
    @staticmethod
    def get_acc(AL, y):
        n = len(AL)
        
        results_binary = np.zeros((n, 1), dtype=np.int32)
        results_binary = (AL == AL.max(axis=1)[:,None]).astype(int)
        
        acc = np.sum(np.all(results_binary == y, axis=1)) / n
        return acc
    
    def run(self, X: np.ndarray, y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray, generations: int, w: float, fi1: float, fi2: float, patience: int, min_delta: float, verbose: float) -> Tuple[List[int], List[int]]:
        """
        Implementation of particle swarm optimization algorithm.
        
        Parameters: 
            X - training set with predictiors
            y - training set with outcome
            val_X - validation set with predictions
            val_y - validation set with outcome
            generations - how many iterations of pso will be executed
            patience - generations without significant improvement, after that learning loop is stopped 
            min_delta - significant improvement between i-th generation cost and (i - patience)-th generation cost
            w - hyperparameter inertia
            fi1 - hyperparameter cognitive(local) acceleration
            fi2 - hyperparameter social(global) acceleration
            verbose - whether to print metrics for every generation
        
        Return:
            * History (list) of train loss
            * History (list) of validation loss 
        """
        val_cost_hist = []
        train_cost_hist = []
        
        n_samples_train = X.shape[0]
        n_samples_val = val_X.shape[0]
        
        # start timing
        t0 = time()
        print_generations_interval = int(generations * .25)

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


            # settings global best particle
            if self.set_gbest_particle():
                gbest_particle_generation = i

            val_cost, val_AL = self.get_cost(self.gbest_particle, val_X, val_y, self.network, self.vector_len)
            train_cost, train_AL = self.get_cost(self.gbest_particle, X, y, self.network, self.vector_len)            
            
            val_acc = PSO.get_acc(val_AL, val_y)
            train_acc = PSO.get_acc(train_AL, y)
            
            average_loss_train = train_cost / n_samples_train
            average_loss_val = val_cost / n_samples_val
            
            val_cost_hist.append(average_loss_val)
            train_cost_hist.append(average_loss_train)

            if patience is not None and min_delta is not None:
                if i - patience >= 0 and train_cost_hist[i - patience] - train_cost_hist[i] <= min_delta or self.gbest_cost[-1] < .01:
                    print(f'Generation {i} stop condition satisfied\nFinal cost: {self.gbest_cost[-1]}')
                    break



            if verbose:
                print(f'Generation {i+1}/{generations}\tloss = {average_loss_train:.5f}\ttrain acc=>{train_acc:.3f}\tval_loss=>{average_loss_val:.5f}\tval acc=>{val_acc:.3f}')
            elif (i+1) % print_generations_interval == 0 or i == 0:
                print(f'Generation {i+1}/{generations}\tloss = {average_loss_train:.5f}\ttrain acc=>{train_acc:.3f}\t val_loss=>{average_loss_val:.5f}\tval acc=>{val_acc:.3f}')


        final_particle = Particle(self.gbest_pos)
        final_particle.fold_vector_to_layer_final(self.network, self.vector_len)

        # stop timing
        t1 = time()
        delta = t1 - t0

        print(f'\nTraining time: {delta:.2f}s')
        return train_cost_hist, val_cost_hist
