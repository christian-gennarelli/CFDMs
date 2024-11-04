from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm
from faiss import IndexFlatL2
from sklearn.neighbors import KNeighborsClassifier


import time
from targets import PointCloud

class Score(ABC):
    # Constructor
    def __init__(self, X):
        self.X = X # Training set
    
    @abstractmethod
    def forward(self, t, z): pass 

class KNN(Score):
    def __init__(self, X, k):
        Score.__init__(self, X)
        self.k = k

        self.index = KNeighborsClassifier(n_neighbors=self.k)
        self.index.fit(
            X,
            np.array(range(0, X.shape[0])),
        )
    
    @abstractmethod
    def forward(t, z): pass

class Faiss(Score):
    def __init__(self, X, k, nprobe):
        Score.__init__(self, X)
        self.k = k # Number of nearest neighbours to search for
        self.nprobe = nprobe # Number of neighbouring Voronoi cells to look at when searching for the nearest neighbours

        D = X.shape[1]
        self.index = IndexFlatL2(D)        
        self.index.add(X)
        self.index.nprobe = nprobe
    
    @abstractmethod
    def forward(self, t, z): pass 

class UnsmoothedScore(Score):
    def __init__(self, X): Score.__init__(self, X)
    
    def forward(self, t, z):
        
        num_samples = self.X.shape[0]
        num_dims = self.X.shape[1]
        
        # Rescale the samples by t
        tX = t*self.X
        # print("tX:\n", tX, '\n')
        assert(tX.shape == (num_samples, num_dims))
        
        # # Calculate distances i.e. z-tX (z is broadcasted to (num_samples, num_dims))
        distance = z-tX
        # print('Distances: \n', distance, '\n')
        assert(distance.shape == (num_samples, num_dims))

        weights = -(norm(distance, axis=1)**2) / (2*(1-t)**2)
        # print('Weights (before softmax): \n', weights, '\n')
        assert(weights.shape == (num_samples,))

        e_x = np.exp(weights - np.max(weights)) # Subtract max value for numerical stability - doesn't affect softmax values
        weights_softmax = e_x/np.sum(e_x, axis=-1, keepdims=True).reshape(-1, 1)
        # print('Weights (after softmax): \n', weights_softmax, '\n')
        assert(weights_softmax.shape == (1, num_samples))
        
        # # Calculate the distance-weighted average of all N rescaled training points
        weighted_sum = np.sum(np.expand_dims(weights_softmax.squeeze(), axis=1) * tX, axis=0)
        # print('Weighted sum: \n', weighted_sum, '\n')
        assert(weighted_sum.shape == (num_dims, ))
        
        # Calculate the score
        score = ((weighted_sum - z) / ((1-t)**2)).squeeze()
        # print("Score (unsmoothed):\n", score, '\n')
        assert(score.shape ==  (num_dims,))

        return score

class SmoothedScore(Score):
    def __init__(self, X, sigma, noise):
        Score.__init__(self, X)
        self.sigma = sigma # Smoothing parameter
        self.noise = noise # Noise to perturb input point
        
    def forward(self, t, z):
        
        num_noises = self.noise.shape[0]
        num_samples = self.X.shape[0]
        num_dims = self.X.shape[1]
        
        # Rescale the samples by t
        tX = t*self.X
        # print('tX:\n', tX, '\n)
        assert(tX.shape == (num_samples, num_dims))

        # Calculate the smoothed noise 
        noise_smoothed = -2*self.sigma*t*np.inner(self.noise, self.X)
        # print('noise_smoothed:\n', noise_smoothed, '\n')
        assert(noise_smoothed.shape == (num_noises, num_samples))

        distance = z-tX
        # print('distance:\n', distance, '\n')
        assert(distance.shape == (num_samples, num_dims))

        # norm_distance = np.sum(np.abs(distance)**2, axis=-1, keepdims=True)
        norm_distance = norm(distance, axis=1, keepdims=True)
        # print('norm_distance:\n', norm_distance, '\n')
        assert(norm_distance.shape == (num_samples, 1))

        weights = -((norm_distance.reshape(-1, num_samples))**2+noise_smoothed) / (2*((1-t)**2))
        # print('weights:\n', weights, '\n')
        assert(weights.shape == (num_noises, num_samples))

        e_x = np.exp(weights - np.max(weights, axis=1, keepdims=True)) # Subtract max value for numerical stability, doesn't affect softmax values
        weights_softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
        # print('weights_softmax:\n', weights_softmax, '\n')
        assert(weights_softmax.shape == (num_noises, num_samples))

        k = np.expand_dims(weights_softmax, axis=2) * tX
        # print('k:\n', k, '\n')
        assert(k.shape == (num_noises, num_samples, num_dims))

        score = (((k.sum(axis=1).sum(axis=0) / num_noises) - z) / ((1-t)**2)).squeeze()
        # print('score:\n', score, '\n')
        assert(score.shape == (num_dims,))

        return score

class UnsmoothedDEANN(UnsmoothedScore, Faiss):
    def __init__(self, X, nprobe, k, l):
        UnsmoothedScore.__init__(self, X)
        Faiss.__init__(self, X, k, nprobe)
        self.l = l
    
    def forward(self, t, z):

        # print('z:\n', z, '\n')
        # print('X:\n', self.X, '\n')

        num_samples, num_dims = self.X.shape
        
        # Search for the nearest neighbours of the input point within the training set
        _, I_nn = self.index.search(z/t, self.k) # Having X (and not tX) stored in the index, I search for the nearest neighbours of z/t instead of z
        nearest_neighbours = np.squeeze(self.X[I_nn])
        # print('nearest_neighbours:\n', nearest_neighbours, '\n')
        assert(nearest_neighbours.shape == (self.k, num_dims))

        # Sample L random samples from the remained of the dataset
        remainder = np.delete(self.X, I_nn, axis=0)
        # print('remainder:\n', remainder, '\n')
        random_samples = np.squeeze(remainder[np.random.choice(np.arange(remainder.shape[0]), size=self.l, replace=False)])
        # print('random_samples:\n', random_samples, '\n')
        assert(random_samples.shape == (self.l, num_dims))

        data = t * np.concatenate((nearest_neighbours, random_samples), axis=0)
        # print('data:\n', data, '\n')
        assert(data.shape == (self.k + self.l, num_dims))

        distance = z - data
        # print('distance:\n', distance, '\n')
        assert(distance.shape == (self.k + self.l, num_dims))

        norm_distance = norm(distance, axis=1, keepdims=True)
        # print('norm_distance:\n', norm_distance, '\n')
        assert(norm_distance.shape == (self.k + self.l, 1))

        weights = -((norm_distance)**2) / (2*((1-t)**2))
        # print('weights:\n', weights, '\n')
        assert(weights.shape == (self.k + self.l, 1))

        max_weight = np.max(weights)
        # print('max_weight:\n', max_weight, '\n')

        e_x = np.exp(weights - max_weight) # Subtract max value for numerical stability, doesn't affect softmax values
        # print('e_x:\n', e_x, '\n')

        # Rescale random samples weights by (N-K)/LN and nearest neighbours by 1/N
        e_x[self.k:, :] *= (num_samples - self.k) / (self.l * num_samples)
        e_x[:self.k, :] *= 1 / num_samples
        #print('e_x rescaled:\n', e_x, '\n')

        weights_softmax = e_x / np.sum(e_x, axis=0, keepdims=True)
        #print('weights_softmax:\n', weights_softmax, '\n')
        assert(weights_softmax.shape == (self.k + self.l, 1))

        k = weights_softmax * data
        # print('k:\n', k, '\n')
        assert(k.shape == (self.k + self.l, num_dims))

        weighted_sum = np.sum(k, axis=0)
        # print('weighted_sum:\n', weighted_sum, '\n')
        assert(weighted_sum.shape == (num_dims,))

        score = ((weighted_sum - z) / ((1-t)**2)).squeeze()
        # print('score:\n', score, '\n')
        assert(score.shape == (num_dims,))

        return score

class SmoothedDEANN(SmoothedScore, Faiss):
    def __init__(self, X, noise, sigma, k, nprobe, l):
        SmoothedScore.__init__(self, X, sigma, noise)
        Faiss.__init__(self, X, k, nprobe)
        self.l = l

    def forward(self, t, z):

        num_samples, num_dims = self.X.shape
        num_noises = self.noise.shape[0]

        # print('z:\n', z, '\n')
        # assert(z.shape == (num_dims, ))

        z_smoothed = z + self.sigma * self.noise
        
        # Search for the nearest neighbours of the input point within the training set
        _, I_nn = self.index.search(z_smoothed/t, self.k) # Having X (and not tX) stored in the index, I search for the nearest neighbours of z/t instead of z
        # print('I_nn:\n', I_nn, '\n')
        assert(I_nn.shape == (num_noises, self.k))
    
        nearest_neighbours = np.squeeze(self.X[I_nn])
        # print('nearest_neighbours:\n', nearest_neighbours, '\n')
        assert(nearest_neighbours.shape == (num_noises, self.k, num_dims))

        X_repeated = np.repeat(self.X[np.newaxis, :, :], I_nn.shape[0], axis=0)

        mask = np.ones(X_repeated.shape, dtype=bool)
        mask[np.arange(I_nn.shape[0])[:, np.newaxis], I_nn] = False
        remainder = np.array([X_repeated[i][mask[i]] for i in range(I_nn.shape[0])]).reshape(I_nn.shape[0], self.X.shape[0] - I_nn.shape[1], self.X.shape[1])
        # print('remainder:\n', remainder, '\n')

        random_indexes = np.array([np.random.choice(np.arange(remainder.shape[1]), size=self.l, replace=False) for _ in range(num_noises)])
        # print('random_indexes:\n', random_indexes, '\n')

        random_samples = remainder[np.arange(remainder.shape[0])[:, None], random_indexes]
        # print('random_samples:\n', random_samples, '\n')
        assert(random_samples.shape == (num_noises, self.l, num_dims))

        data = t * np.concatenate((nearest_neighbours, random_samples), axis=1)
        # print('data:\n', data, '\n')
        assert(data.shape == (num_noises, self.k + self.l, num_dims))

        distances = z_smoothed[:, np.newaxis, :] - data
        # print('distances:\n', distances, '\n')
        assert(distances.shape == (num_noises, self.k + self.l, num_dims))

        norm_distance = norm(distances, axis=2, keepdims=True)
        # print('norm_distance:\n', norm_distance, '\n')
        assert(norm_distance.shape == (num_noises, self.k + self.l, 1))

        weights = -((norm_distance)**2) / (2*((1-t)**2))
        # print('weights:\n', weights, '\n')
        assert(weights.shape == (num_noises, self.k + self.l, 1))

        max_weight = np.max(weights, axis=1)
        # print('max_weight:\n', max_weight, '\n')
        assert(max_weight.shape == (num_noises, 1))

        e_x = np.exp(weights - max_weight[:, np.newaxis, :]) # Subtract max value for numerical stability, doesn't affect softmax values
        # print('e_x:\n', e_x, '\n')
        assert(e_x.shape == (num_noises, self.k + self.l, 1))

        # # Rescale random samples weights by (N-K)/LN and nearest neighbours by 1/N
        e_x[:, self.k:, :] *= (num_samples - self.k) / (self.l * num_samples)
        e_x[:, :self.k, :] *= 1 / num_samples
        # #print('e_x rescaled:\n', e_x, '\n')

        weights_softmax = e_x / np.sum(e_x, axis=1, keepdims=True)
        # print('weights_softmax:\n', weights_softmax, '\n')
        assert(weights_softmax.shape == (num_noises, self.k + self.l, 1))

        k = weights_softmax * data
        # print('k:\n', k, '\n')
        assert(k.shape == (num_noises, self.k + self.l, num_dims))

        weighted_sum = np.sum(np.sum(k, axis=1), axis=0) / num_noises
        # print('weighted_sum:\n', weighted_sum, '\n')
        assert(weighted_sum.shape == (num_dims,))

        score = ((weighted_sum - z) / ((1-t)**2)).squeeze()
        # print('score:\n', score, '\n')
        assert(score.shape == (num_dims,))

        return score
    
######### TESTING #########
if __name__ == '__main__':
    pass
###########################