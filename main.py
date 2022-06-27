from GDR import GravitionalDimensionalityReduction
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def main():
    # load dataset:
    (D, labels) = load_digits(return_X_y=True)
    print(D.shape)
    D = D[:50, :]
    # plt.matshow(D[0, :].reshape((8,8)))
    # plt.gray()
    # plt.show()

    # instantiate class:
    gdr = GravitionalDimensionalityReduction(max_itrations=10, alpha=1, final_DR_method=None)
    D_transformed = gdr.fit_transform(D=D, labels=None)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()