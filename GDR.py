import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
import math
import utils
from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt

DEBUG_MODE = True
VERBOSITY = 2
SHOW_VISUALIZATION = True

class GravitionalDimensionalityReduction():
    def __init__(self, max_itrations=100, alpha=1, final_DR_method=None, supervised_mode=False, do_sort_by_density=True, method='Newtonian') -> None:
        self._max_itrations = max_itrations
        self._alpha = alpha
        if final_DR_method is None:
            self._final_DR_method = PCA()
        else:
            self._final_DR_method = final_DR_method
        self._supervised_mode = supervised_mode
        self._do_sort_by_density = do_sort_by_density
        self._method = method
        self._n_samples = None
        self._dimensionality = None
        self._n_classes = None
        self._class_names = None

    def fit_transform(self, D: np.ndarray, labels: Optional[np.array] = None):
        """
        Fit and transform the data for dimensionality reduction.

        Args:
            D (np.ndarray): The row-wise dataset, with rows as samples and columns as features
            labels (np.array): The labels of samples, if the samples are labeled. 
        """
        if self._supervised_mode:
            assert(labels is not None)

        # make D column-wise:
        D = D.T

        # paremeters:
        self._n_samples = D.shape[1]
        self._dimensionality = D.shape[0]
        self._n_classes = len(np.unique(labels))
        self._class_names = [str(i) for i in range(self._n_classes)]

        # apply PCA to go to PCA subspace (space manifold in physics):
        pca = PCA(n_components=3)
        X = (pca.fit_transform(D.T)).T
        
        # in supervised case, make X_classes from X:
        if self._supervised_mode:
            X_classes, indices_classes = self._convert_X_to_classes(X, labels)
        else:
            X_classes, indices_classes = None, None

        # sort based on density:
        if self._do_sort_by_density:
            if not self._supervised_mode:
                X, labels, sorted_indices = self._sort_by_density(X=X, labels=labels)
            else:
                sorted_indices = [None] * self._n_classes
                for label in range(self._n_classes):
                    X_classes[label], sorted_indices[label] = self._sort_by_density(X=X_classes[label])
        else:
            sorted_indices = None

        if DEBUG_MODE and SHOW_VISUALIZATION: 
            self._visualize_embedding(X=X, X_classes=X_classes, labels=labels, sorted_indices=sorted_indices, indices_classes=indices_classes)

        # iterations of algorithm:
        for itr in range(self._max_itrations):
            if DEBUG_MODE: print(f'===== iteration: {itr}')
            if not self._supervised_mode:
                if self._method == 'Newtonian':
                    X = self._main_algorithm_Newtonian(X=X)
                elif self._method == 'Relativity':
                    X = self._main_algorithm_Relativity(X=X)
            else:
                for label in range(self._n_classes):
                    if self._method == 'Newtonian':
                        X_classes[label] = self._main_algorithm_Newtonian(X=X_classes[label])
                    elif self._method == 'Relativity':
                        X_classes[label] = self._main_algorithm_Relativity(X=X_classes[label])

        if DEBUG_MODE and SHOW_VISUALIZATION:
            self._visualize_embedding(X=X, X_classes=X_classes, labels=labels, sorted_indices=sorted_indices, indices_classes=indices_classes)

        # in supervised case, make X from X_classes:
        if self._supervised_mode:
            X = self._convert_classes_to_X(X_classes, indices_classes)

        # reconstruct from PCA subspace (space manifold in physics):
        D_modified = pca.inverse_transform(X=X.T)  # NOTE: D_modified is row-wise

        # apply any dimensionality reduction method on the modified D:
        D_transformed = self._final_DR_method.fit_transform(D_modified)  # NOTE: D_transformed is row-wise

        return D_transformed

    def _main_algorithm_Newtonian(self, X):
        n_samples = X.shape[1]
        for j in range(1, n_samples+1):  # affected by the gravitation of particles
            if DEBUG_MODE and VERBOSITY >= 2: 
                if j % 50 == 0:
                    print(f'Processing instance {j} / {n_samples}')
            x_j = X[:, -j]
            delta = 0
            for i in range(n_samples):  # the particle having gravity
                x_i = X[:, i]
                if i == (n_samples-j): continue         
                if np.all(x_j == x_i): continue
                r_ij = np.linalg.norm(x_i - x_j)
                delta_ij_value = 1/r_ij
                delta_ij_direction = x_i - x_j
                delta_ij = delta_ij_value * delta_ij_direction
                delta += delta_ij
            x_j = x_j + delta
            X[:, -j] = x_j
        return X

    def _main_algorithm_Relativity(self, X):
        n_samples = X.shape[1]
        for j in range(1, n_samples+1):  # affected by the gravitation of particles
            if DEBUG_MODE and VERBOSITY >= 2: 
                if j % 50 == 0:
                    print(f'Processing instance {j} / {n_samples}')
            x_j = X[:, -j]
            for i in range(n_samples):  # the particle having gravity
                x_i = X[:, i]
                if i == (n_samples-j): continue         
                if np.all(x_j == x_i): continue
                # calculate r:
                r_ij = np.linalg.norm(x_i - x_j)
                # weights:
                beta = [0.3, 0.3, 0.3]
                # beta = [0.8, 0.1, 0.1]
                # beta = [1, 0, 0]
                beta = beta / np.sum(beta)
                assert (np.sum(beta) == 1)
                # amount of movement:
                movement_amount = 1/r_ij
                movement_amount_in_r = movement_amount * beta[0]
                movement_amount_in_theta = movement_amount * beta[1]
                movement_amount_in_phi = movement_amount * beta[2]
                # parameters:
                G, M, c = 1, 1, 1
                r, theta = self._caculate_r_and_theta(origin=x_i, x=x_j)
                # movement in r:
                temp = 1 - ((2 * G * M) / (r * (c**2)))
                r_component = 1 / temp
                delta_ij_value_r = -1 * (movement_amount_in_r / r_component)**0.5
                # movement in theta:
                theta_component = r**2
                delta_ij_value_theta = -1 * (movement_amount_in_theta / theta_component)**0.5
                # movement in phi:
                phi_component = (r**2) * (math.sin(theta)**2)
                delta_ij_value_phi = -1 * (movement_amount_in_phi / phi_component)**0.5
                # movement:
                delta_ij = [delta_ij_value_r, delta_ij_value_theta, delta_ij_value_phi]
                x_k = self._move_in_spherical_coordinate_system(x=x_j, origin=x_i, delta=delta_ij)

                r_ij, theta_ij = self._caculate_r_and_theta(origin=x_i, x=x_j)
                M_ij = self._alpha / np.linalg.norm(x_i - x_j)
                # M_ij = self._alpha * np.linalg.norm(x_i - x_j)
                g_ij = self._Schwarzschild_metric(r=r_ij, theta=theta_ij, M=M_ij, G=1, c=1, ignore_time_component=True)
                # print(g_ij[1,1], r_ij, theta_ij, M_ij, x_i, x_j)
                eig_vectors, eig_values = self._solve_eigenvalue_problem(matrix=g_ij, sort=True, sort_descending=True, n_components=None)
                # delta_ij = eig_vectors[:, 0] * eig_values[0]
                delta_ij = eig_vectors[:, 0]
                # import pdb; pdb.set_trace()
                # if not np.all(eig_vectors == np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])):
                #     print(eig_vectors)
                #     import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                x_j = self._move_in_spherical_coordinate_system(x=x_j, origin=x_i, delta=delta_ij)
                # print(x_j, x_i)
            X[:, -j] = x_j
        return X

    # def _main_algorithm_Relativity(self, X):
    #     n_samples = X.shape[1]
    #     for j in range(1, n_samples+1):  # affected by the gravitation of particles
    #         if DEBUG_MODE and VERBOSITY >= 2: 
    #             if j % 50 == 0:
    #                 print(f'Processing instance {j} / {n_samples}')
    #         x_j = X[:, -j]
    #         # print('------------------------------')
    #         for i in range(n_samples):  # the particle having gravity
    #             x_i = X[:, i]
    #             if i == (n_samples-j): continue         
    #             if np.all(x_j == x_i): continue
    #             r_ij, theta_ij = self._caculate_r_and_theta(origin=x_i, x=x_j)
    #             M_ij = self._alpha / np.linalg.norm(x_i - x_j)
    #             # M_ij = self._alpha * np.linalg.norm(x_i - x_j)
    #             g_ij = self._Schwarzschild_metric(r=r_ij, theta=theta_ij, M=M_ij, G=1, c=1, ignore_time_component=True)
    #             # print(g_ij[1,1], r_ij, theta_ij, M_ij, x_i, x_j)
    #             eig_vectors, eig_values = self._solve_eigenvalue_problem(matrix=g_ij, sort=True, sort_descending=True, n_components=None)
    #             # delta_ij = eig_vectors[:, 0] * eig_values[0]
    #             delta_ij = eig_vectors[:, 0]
    #             # import pdb; pdb.set_trace()
    #             # if not np.all(eig_vectors == np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])):
    #             #     print(eig_vectors)
    #             #     import pdb; pdb.set_trace()
    #             # import pdb; pdb.set_trace()
    #             x_j = self._move_in_spherical_coordinate_system(x=x_j, origin=x_i, delta=delta_ij)
    #             # print(x_j, x_i)
    #         X[:, -j] = x_j
    #     return X

    def _move_in_spherical_coordinate_system(self, x, origin, delta):
        assert (not np.all(x == origin))
        # shift based on origin:
        x = x - origin
        # Cartesian to spherical conversion:
        x_spherical = self._convert_Cartesian_to_spherical_coordinates(x=x)
        # movement in spherical coordinate system:
        x_spherical = x_spherical + delta
        # print(x, origin, x_spherical, delta)
        # spherical to Cartesian conversion:
        x = self._convert_spherical_to_Cartesian_coordinates(x=x_spherical)
        # shift back based on origin:
        x = x + origin
        return x

    def _convert_Cartesian_to_spherical_coordinates(self, x):
        """
        Convert the point coordinates in Cartesian coordinate system to the point coordinates in the pherical coordinate system.

        Notes:
            https://en.wikipedia.org/wiki/Spherical_coordinate_system
            Cartesian to spherical conversion (but in this link, the notations of theta and phi are replaced.): 
                https://keisan.casio.com/exec/system/1359533867
        """
        # Cartesian to spherical conversion (calculate r):
        r = np.linalg.norm(x)
        # Cartesian to spherical conversion (calculate theta):
        r_in_x_y_plane = (x[0]**2 + x[1]**2) ** 0.5
        theta = np.arctan(r_in_x_y_plane / np.abs(x[2]))
        if x[2] < 0:
            theta = np.pi - theta
        # Cartesian to spherical conversion (calculate phi):
        phi = np.arctan(np.abs(x[1]) / np.abs(x[0]))
        if x[0] >= 0 and x[1] >= 0:
            pass
        elif x[0] < 0 and x[1] >= 0:
            phi = np.pi - phi
        elif x[0] < 0 and x[1] < 0:
            phi = np.pi + phi
        elif x[0] >= 0 and x[1] < 0:
            phi = (2*np.pi) - phi
        assert (not np.isnan(np.asarray([r, theta, phi])).any())
        return np.asarray([r, theta, phi])

    def _convert_spherical_to_Cartesian_coordinates(self, x):
        """
        Convert the point coordinates in spherical coordinate system to the point coordinates in Cartesian coordinate system.

        Notes:
            https://en.wikipedia.org/wiki/Spherical_coordinate_system
            Cartesian to spherical conversion (but in this link, the notations of theta and phi are replaced.): 
                https://keisan.casio.com/exec/system/1359534351
        """
        r, theta, phi = x
        # spherical to Cartesian conversion (calculate x):
        x = r * np.sin(theta) * np.cos(phi)
        # spherical to Cartesian conversion (calculate y):
        y = r * np.sin(theta) * np.sin(phi)
        # spherical to Cartesian conversion (calculate z):
        z = r * np.cos(theta)
        assert (not np.isnan(np.asarray([x, y, z])).any())
        return np.asarray([x, y, z])

    def _caculate_r_and_theta(self, origin, x):
        """
        Calculate r and theta in the spherical coordinate system with a specified origin.

        Notes:
            https://en.wikipedia.org/wiki/Spherical_coordinate_system
        """
        x = x - origin
        # calculate r:
        r = np.linalg.norm(x)
        # calculate theta:
        r_in_x_y_plane = (x[0]**2 + x[1]**2) ** 0.5
        theta = np.arctan(r_in_x_y_plane / np.abs(x[2]))
        if x[2] < 0:
            theta = np.pi - theta
        return (r, theta)

    def _Schwarzschild_metric(self, r, theta, M, G=1, c=1, ignore_time_component=False):
        temp = 1 - ((2 * G * M) / (r * (c**2)))
        if not ignore_time_component:
            t_component = temp
        r_component = (1 / temp)
        theta_component = r**2
        phi_component = (r**2) * (math.sin(theta)**2)
        if not ignore_time_component:
            metric = np.diag([t_component, r_component, theta_component, phi_component])
        else:
            metric = np.diag([r_component, theta_component, phi_component])
        return metric

    def _solve_eigenvalue_problem(self, matrix, sort=True, sort_descending=True, n_components=None):
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

    def _sort_by_density(self, X, labels=None):
        """
        Sort the samples based on density of Local Outlier Factor (LOF).

        Args:
            X (np.ndarray): The column-wise dataset, with columns as samples and rows as features

        Note:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
            https://en.wikipedia.org/wiki/Local_outlier_factor
        """
        lof = LOF(n_neighbors=10)
        lof.fit(X.T)
        density_scores = lof.negative_outlier_factor_
        # density_scores = lof.score_samples(X.T)
        # sort from largest to smallest score:
        sorted_indices = np.argsort(density_scores)[::-1]
        X = X[:, sorted_indices]
        if labels is not None:
            labels = labels[sorted_indices]
            return X, labels, sorted_indices
        else:
            return X, sorted_indices

    def _unsort(self, sorted_indices, X, labels=None):
        X_unsorted = np.zeros_like(X)
        if labels is not None:
            labels_unsorted = np.zeros_like(labels)
        for i in range(X.shape[1]):
            X_unsorted[:, sorted_indices[i]] = X[:, i]
            if labels is not None:
                labels_unsorted[sorted_indices[i]] = labels[i]
        if labels is not None:
            return X_unsorted, labels_unsorted
        else:
            return X_unsorted

    def _convert_X_to_classes(self, X, labels):
        X_classes, indices_classes = [], []
        n_classes = len(np.unique(labels))
        for label in range(n_classes):
            condition = (labels == label)
            indices = np.array([int(i) if condition[i] else np.nan for i in range(len(condition))])
            indices = indices[~np.isnan(indices)]
            indices = indices.astype(int)
            X_class = X[:, indices].copy()
            X_classes.append(X_class)
            indices_classes.append(indices)
        return X_classes, indices_classes

    def _convert_classes_to_X(self, X_classes, indices_classes):
        n_classes = len(X_classes)
        n_samples = 0
        n_dimensions = X_classes[0].shape[0]
        for label in range(n_classes):
            X_class = X_classes[label]
            n_samples += X_class.shape[1]
        X = np.zeros((n_dimensions, n_samples))
        for label in range(n_classes):
            X[:, indices_classes[label]] = X_classes[label]
        return X

    def _visualize_embedding(self, X, X_classes, labels, sorted_indices, indices_classes):
        if not self._supervised_mode:
            if self._do_sort_by_density:
                X_plot, labels_plot = self._unsort(sorted_indices, X, labels)
            else:
                X_plot, labels_plot = X.copy(), labels.copy()
        else:
            if self._do_sort_by_density:
                X_classes_unsorted = X_classes.copy()
                for label in range(self._n_classes):
                    X_classes_unsorted[label] = self._unsort(sorted_indices=sorted_indices[label], X=X_classes[label])
                X_plot = self._convert_classes_to_X(X_classes_unsorted, indices_classes)
            else:
                X_plot = self._convert_classes_to_X(X_classes, indices_classes)
            labels_plot = labels.copy()
        plt = utils.plot_3D(X=X_plot.T, labels=labels_plot, class_names=self._class_names)
        plt.show()

    def test_Newtonian_movement(self):
        x_i = np.array([1, 1, 1])
        x_j = np.array([2, 3, 4])
        r_ij = np.linalg.norm(x_i - x_j)
        delta_ij_value = 1/r_ij
        delta_ij_direction = x_i - x_j
        delta_ij = delta_ij_value * delta_ij_direction
        x_k = x_j + delta_ij

        labels = [0,1,2]
        self._n_classes = len(np.unique(labels))
        self._class_names = [str(i) for i in range(self._n_classes)]
        plt = utils.plot_3D(X=np.asarray([x_i, x_j, x_k]), labels=labels, class_names=self._class_names)
        plt.show()

    def test_Relativity_movement(self):
        x_i = np.array([1, 1, 1])
        x_j = np.array([2, 3, 4])
        r_ij = np.linalg.norm(x_i - x_j)
        # weights:
        beta = [0.3, 0.3, 0.3]
        # beta = [0.8, 0.1, 0.1]
        # beta = [1, 0, 0]
        beta = beta / np.sum(beta)
        assert (np.sum(beta) == 1)
        # amount of movement:
        movement_amount = 1/r_ij
        movement_amount_in_r = movement_amount * beta[0]
        movement_amount_in_theta = movement_amount * beta[1]
        movement_amount_in_phi = movement_amount * beta[2]
        # calculate r and theta:
        r, theta = self._caculate_r_and_theta(origin=x_i, x=x_j)
        # tensor components:
        g = self._Schwarzschild_metric(r=r, theta=theta, M=1, G=1, c=1, ignore_time_component=True)
        r_component = g[0, 0]
        theta_component = g[1, 1]
        phi_component = g[2, 2]
        # movement in r, theta, and phi directions:
        delta_ij_value_r = -1 * (movement_amount_in_r / r_component)**0.5
        delta_ij_value_theta = -1 * (movement_amount_in_theta / theta_component)**0.5
        delta_ij_value_phi = -1 * (movement_amount_in_phi / phi_component)**0.5
        # overall movement:
        delta_ij = [delta_ij_value_r, delta_ij_value_theta, delta_ij_value_phi]
        x_k = self._move_in_spherical_coordinate_system(x=x_j, origin=x_i, delta=delta_ij)

        labels = [0,1,2]
        self._n_classes = len(np.unique(labels))
        self._class_names = [str(i) for i in range(self._n_classes)]
        plt = utils.plot_3D(X=np.asarray([x_i, x_j, x_k]), labels=labels, class_names=self._class_names)
        plt.show()