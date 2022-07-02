from GDR import GravitionalDimensionalityReduction
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def main():
    # load dataset:
    n_samples = 1000
    (D, labels) = load_digits(return_X_y=True)
    D = D[:n_samples, :]
    labels = labels[:n_samples]
    print(D.shape)
    # plt.matshow(D[0, :].reshape((8,8)))
    # plt.gray()
    # plt.show()

    # instantiate class:
    gdr = GravitionalDimensionalityReduction(max_itrations=5, alpha=1, final_DR_method=None, supervised_mode=True, do_sort_by_density=True)
    # gdr.test()
    D_transformed = gdr.fit_transform(D=D, labels=labels)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()