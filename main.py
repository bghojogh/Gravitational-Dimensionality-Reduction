from GDR import GravitionalDimensionalityReduction
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# parameters of dataset:
N_SAMPLES = None  #--> None, 1000, ...

# parameters of class:
MAX_ITERATIONS = 10
ALPHA = [0.33, 0.33, 0.33]  #--> [0.33, 0.33, 0.33], [1, 0, 0], ...
SUPERVISED_MODE = True
DO_SORT_BY_DENSITY = True
METHOD = 'Relativity'  #--> 'Newtonian', 'Relativity'
METRIC = 'Schwarzschild'  #--> 'Schwarzschild', 'Minkowski'
USE_PCA_FOR_NEWTONIAN = False

# experiment type:
EXPERIMENT = 'main_algorithm' #--> 'main_algorithm', 'test_Newtonian_movement', 'test_Relativity_movement'

def main():
    # load dataset:
    (D, labels) = load_digits(return_X_y=True)
    if N_SAMPLES is not None:
        D = D[:N_SAMPLES, :]
        labels = labels[:N_SAMPLES]
    
    # instantiate class:
    gdr = GravitionalDimensionalityReduction(max_itrations=MAX_ITERATIONS, alpha=ALPHA, supervised_mode=SUPERVISED_MODE, 
                                            do_sort_by_density=DO_SORT_BY_DENSITY, method=METHOD, metric=METRIC,
                                            use_PCA_for_Newtonian=USE_PCA_FOR_NEWTONIAN)
    
    # experiment:
    if EXPERIMENT == 'main_algorithm':
        D_transformed = gdr.fit_transform(D=D, labels=labels)
    elif EXPERIMENT == 'test_Newtonian_movement':
        gdr.test_Newtonian_movement()
    elif EXPERIMENT == 'test_Relativity_movement':
        gdr.test_Relativity_movement()

def display_data(D, sample_index):
    print(f'Size of dataset: {D.shape}')
    plt.matshow(D[sample_index, :].reshape((8,8)))
    plt.gray()
    plt.show()

if __name__ == "__main__":
    main()