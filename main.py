import numpy as np
from sklearn.decomposition import PCA

from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

import matplotlib.pyplot as plt

plt.matshow(digits.images[0])
plt.gray()
plt.show()
