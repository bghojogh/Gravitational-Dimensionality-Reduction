import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE
from typing import Optional, List

def plot_3D(X, labels, class_names):
    figure_size = (12, 12)  #--> example: (14, 10)
    dot_sizes = 50  #--> example: 10
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=dot_sizes, c=labels, cmap='Spectral', alpha=1.0)
    return plt

def plot_embedding_of_points(embedding: np.ndarray, labels: Optional[np.array], class_names: Optional[List[str]], 
                            n_samples_plot: Optional[int] = None, method: Optional[str] = 'tsne'):
    """
    Plot the embedding for visualization.

    Args:
        embedding (np.ndarray): The row-wise embedding dataset, with rows as samples and columns as features
        labels (np.array): The labels of samples, if the samples are labeled.
        class_names (List[str]): the names of classes, if the samples are labeled.
        n_samples_plot (int): the number of samples to plot. It is optional. If not set, all points are plotted.
        method (str): the method for visualization, if the dimensionlaity of embedding is not 2. 
            Default is 'tsne'. Options are 'tsne' and 'umap'.
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
    n_classes = len(class_names)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=dot_sizes, c=labels, cmap='Spectral', alpha=1.0)
    # plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
    # cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.7)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(class_names)
    return plt