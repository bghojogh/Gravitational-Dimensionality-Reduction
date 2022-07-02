from tkinter.tix import MAX
from GDR import GravitionalDimensionalityReduction
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# parameters of dataset:
N_SAMPLES = 1000

# parameters of class:
MAX_ITERATIONS = 5
ALPHA = 1
FINAL_DR_METHOD = None
SUPERVISED_MODE = True
DO_SORT_BY_DENSITY = True
METHOD = 'Newtonian'  #--> 'Newtonian', 'Relativity'

# experiment type:
EXPERIMENT = 'test_Relativity_movement' #--> 'main_algorithm', 'test_Newtonian_movement', 'test_Relativity_movement'

def main():
    # load dataset:
    (D, labels) = load_digits(return_X_y=True)
    D = D[:N_SAMPLES, :]
    labels = labels[:N_SAMPLES]
    
    # instantiate class:
    gdr = GravitionalDimensionalityReduction(max_itrations=MAX_ITERATIONS, alpha=ALPHA, final_DR_method=FINAL_DR_METHOD, 
                                            supervised_mode=SUPERVISED_MODE, do_sort_by_density=DO_SORT_BY_DENSITY, method=METHOD)
    
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