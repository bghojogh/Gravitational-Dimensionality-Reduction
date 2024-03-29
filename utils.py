import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE
from typing import Optional, List, Tuple, TypeVar
matplotlib_pyplot = TypeVar('matplotlib_pyplot')

def save_variable(variable, name_of_variable, path_to_save='./'):
    """
    Save variable to computer.

    Notes: 
        https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    """
    if not os.path.exists(path_to_save):  
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    """
    Load variable from computer.

    Notes: 
        https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    """
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    """
    Save np.ndarray variable to a text file in computer.

    Notes: 
        https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    """
    if type(variable) is list:
        variable = np.asarray(variable)
    if not os.path.exists(path_to_save):  
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def plot_3D(X: np.ndarray, labels: np.array, class_names: List[str]) -> matplotlib_pyplot:
    """
    Visualize data in 3D plot.

    Args:
        X (np.ndarray): the column-wise dataset, with columns as samples and rows as features 
        labels (np.array): the labels of samples, if the samples are labeled
        class_names (List[str]): the names of classes, with the order of labels.

    Returns:
        plt (matplotlib.pyplot): the plot object. Use plt.show or plt.savefig for showing or saving it, respectively.
    """
    figure_size = (12, 12)  #--> example: (14, 10)
    dot_sizes = 50  #--> example: 10
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(projection='3d')
    p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=dot_sizes, c=labels, cmap='Spectral', alpha=1.0)
    fig.colorbar(p)
    return plt

def plot_embedding_of_points(embedding: np.ndarray, labels: Optional[np.array], class_names: Optional[List[str]] = None, 
                            n_samples_plot: Optional[int] = None, method: Optional[str] = 'tsne') -> matplotlib_pyplot:
    """
    Plot the embedding for visualization.

    Args:
        embedding (np.ndarray): The row-wise embedding dataset, with rows as samples and columns as features
        labels (np.array): The labels of samples, if the samples are labeled.
        class_names (List[str]): the names of classes, if the samples are labeled.
        n_samples_plot (int): the number of samples to plot. It is optional. If not set, all points are plotted.
        method (str): the method for visualization, if the dimensionlaity of embedding is not 2. 
            Default is 'tsne'. Options are 'tsne' and 'umap'.

    Returns:
        plt (matplotlib.pyplot): the plot object. Use plt.show or plt.savefig for showing or saving it, respectively.
    """
    figure_size = (7, 5)  #--> example: (14, 10)
    dot_sizes = 50  #--> example: 10
    n_samples, n_dimensions = embedding.shape
    if n_samples_plot != None:
        indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
        embedding = embedding[indices_to_plot, :]
        labels = labels[indices_to_plot]
    if n_dimensions != 2:
        if method == 'umap':
            embedding = umap.UMAP(n_neighbors=500).fit_transform(embedding)
        elif method == 'tsne':
            embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embedding)
    _, ax = plt.subplots(1, figsize=figure_size)
    if class_names is None:
        n_classes = len(np.unique(labels))
        class_names = [str(i) for i in range(n_classes)]
    n_classes = len(class_names)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=dot_sizes, c=labels, cmap='Spectral', alpha=1.0)
    # plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
    # cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.7)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(class_names)
    return plt

def solve_eigenvalue_problem(matrix: np.ndarray, sort: Optional[bool] = True, sort_descending: Optional[bool] = True, 
                            n_components: Optional[int] = None) -> Tuple[np.ndarray, np.array]:
    """
    Solve the eigenvalue problem for the input matrix.

    Args:
        matrix (np.ndarray): the input matrix
        sort (bool): whether to sort the eigenvalues (and hence eigenvectors) or not
        sort_descending (bool): If true, sorting is descending; otherwise, it is ascending
        n_components (int): the number of eigenvectors and eigenvalues to return. If None, all of them are returned.

    Returns:
        eig_vec (np.ndarray): the matrix of eigenvectors. Every column is an eigenvector.
        eig_val (np.array): the vector containing eigenvalues.
    """
    eig_val, eig_vec = np.linalg.eigh(matrix)
    if sort:
        if sort_descending:
            idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        else:
            idx = eig_val.argsort()  # sort eigenvalues in ascending order (smallest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
    if n_components is not None:
        eig_vec = eig_vec[:, 0:n_components]
    else:
        eig_vec = eig_vec
    return eig_vec, eig_val