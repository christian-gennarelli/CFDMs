import sys
sys.path.insert(0, " thesis") 

import time

import faiss
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from targets import PointCloud

def main():

    '''
    Experiment showing the difference in terms of running times
    between using FAISS and simple KNN.
    '''

    num_samples = 100000
    X = PointCloud('car_body').sample(num_samples)
    y = np.array(range(0, X.shape[0]))

    query = np.random.multivariate_normal(np.zeros(3), np.eye(3), 1)
    print(f'Query vector: {query}\n')

    start_knn = time.time()
    knn = KNeighborsClassifier(
        n_neighbors = 1
    )
    knn.fit(X, y)
    nn_idx_knn = knn.predict(query)
    # print(X[nn_idx_knn], '\n')
    end_knn = time.time()
    print(f'KNN runtime: {end_knn - start_knn}')

    start_faiss = time.time()
    index = faiss.IndexFlatL2(3)
    index.add(X)
    index.nprobe = 5
    _, nn_idx_faiss = index.search(query, 1)
    # print(X[nn_idx_faiss])
    end_faiss = time.time()
    print(f'FAISS runtime: {end_faiss - start_faiss}')

    ### FAISS WINS (of course...) ### 
    # KNN runtime: 0.03831791877746582
    # FAISS runtime: 0.0016739368438720703

if __name__ == '__main__':
    main()