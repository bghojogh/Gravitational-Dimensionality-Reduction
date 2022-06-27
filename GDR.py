import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
import math

DEBUG_MODE = True
VERBOSITY = 1

class GravitionalDimensionalityReduction():
    def __init__(self, max_itrations=100, alpha=1, final_DR_method=None) -> None:
        self._max_itrations = max_itrations
        self._alpha = alpha
        if final_DR_method is None:
            self._final_DR_method = PCA()
        else:
            self._final_DR_method = final_DR_method
        self._n_samples = None
        self._dimensionality = None
        self._n_classes = None

    def fit_transform(self, D: np.ndarray, labels: Optional[np.array] = None):
        """
        Fit and transform the data for dimensionality reduction.

        Args:
            D (np.ndarray): The row-wise dataset, with rows as samples and columns as features
            labels (np.array): The labels of samples, if the samples are labeled. 
        """
        # make D column-wise:
        D = D.T

        # paremeters:
        self._n_samples = D.shape[1]
        self._dimensionality = D.shape[0]
        self._n_classes = len(np.unique(labels))

        # apply PCA to go to PCA subspace (space manifold in physics):
        pca = PCA(n_components=3)
        X = (pca.fit_transform(D.T)).T

        # sort based on density:
        if labels is None:
            X = self._sort_by_density(X=X)
        else:
            X_classes = []
            for label in range(self._n_classes):
                X_class = X[:, labels == label].copy()
                X_class = self._sort_by_density(X=X_class, class_label=label)
                X_classes.append(X_class)

        # iterations of algorithm:
        for itr in range(self._max_itrations):
            if DEBUG_MODE: print(f'===== iteration: {itr}')
            if labels is None:
                X = self._main_algorithm(X=X)
            else:
                for label in range(self._n_classes):
                    X_classes = self._main_algorithm(X=X_classes[label])
                
        if labels is not None:
            # TODO: make X from X_classes
            pass

        # reconstruct from PCA subspace (space manifold in physics):
        D_modified = pca.inverse_transform(X=X.T)  # NOTE: D_modified is row-wise

        # apply any dimensionality reduction method on the modified D:
        D_transformed = self._final_DR_method.fit_transform(D_modified)  # NOTE: D_transformed is row-wise

        return D_transformed
                    
    def _main_algorithm(self, X):
        n_samples = X.shape[1]
        for j in range(n_samples):  # affected by the gravitation of particles
            if DEBUG_MODE and VERBOSITY >= 2: print(f'Processing instance {j} / {n_samples}')
            x_j = X[:, -j-1]
            for i in range(n_samples):  # the particle having gravity
                if i == j: continue
                x_i = X[:, i]
                r_ij, theta_ij = self._caculate_r_and_theta(origin=x_i, x=x_j)
                M_ij = self._alpha * np.linalg.norm(x_i - x_j)
                g_ij = self._Schwarzschild_metric(r=r_ij, theta=theta_ij, M=M_ij, G=1, c=1, ignore_time_component=True)
                delta_ij = self._solve_eigenvalue_problem(matrix=g_ij, sort_descending=True, n_components=None)
                x_j = self._move_in_spherical_coordinate_system(x=x_j, origin=x_i, delta=delta_ij)
            X[:, -j-1] = x_j
        return X

    def _move_in_spherical_coordinate_system(self, x, origin, delta):
        # TODO: to be implemented
        return x

    def _Schwarzschild_metric(self, r, theta, M, G=1, c=1, ignore_time_component=False):
        temp = 1 - ((2 * G * M) / (r * (c**2)))
        if not ignore_time_component:
            t_component = - temp
        r_component = 1 / temp
        theta_component = r**2
        phi_component = (r**2) * (math.sin(theta)**2)
        if not ignore_time_component:
            metric = np.diag([t_component, r_component, theta_component, phi_component])
        else:
            metric = np.diag([r_component, theta_component, phi_component])
        return metric

    def _solve_eigenvalue_problem(self, matrix, sort_descending=True, n_components=None):
        eig_val, eig_vec = np.linalg.eigh(matrix)
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
        return eig_vec

    def _caculate_r_and_theta(self, origin, x):
        # TODO: to be implemented
        r = 0
        theta = 0
        return (r, theta)

    def _sort_by_density(self, X, label=None):
        """
        Sort the samples based on density of Local Outlier Factor (LOF).

        Note:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
            https://en.wikipedia.org/wiki/Local_outlier_factor
        """
        # TODO: to be implemented
        return X