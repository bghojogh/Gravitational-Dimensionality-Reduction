import numpy as np
from sklearn.decomposition import PCA
import math
import utils
from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Union, Tuple, TypeVar
matplotlib_pyplot = TypeVar('matplotlib_pyplot')

DEBUG_MODE = True
VERBOSITY = 2
SHOW_VISUALIZATION = False
SAVE_VISUALIZATION = True
PATH_SAVE = './saved_files/'

class GravitionalDimensionalityReduction():
    def __init__(self, max_itrations: Optional[int] = 100, alpha: Optional[List[float]] = [0.33, 0.33, 0.33], 
                supervised_mode: Optional[bool] = False, do_sort_by_density: Optional[bool] = True, 
                method: Optional[str] = 'Relativity', metric: Optional[str] = "Schwarzschild", 
                use_PCA_for_Newtonian: Optional[bool] = False) -> None:
        """Class for Gravitational Dimensionality Reduction (GDR)
        
        Args:
            max_itrations (int): the number of iterations for GDR algorithm
            alpha (List[float]): the weights of movements in directions of every component in the space manifold.
                The summation of elements of this list should be one.
                This variable is only used for the Relativity method and not the Newtonian method.
            supervised_mode (bool): if true, it is supervised; otherwise, it is unsupervised. 
                The supervised version of GDR works much better. The unsupervised version is work in progress. 
            do_sort_by_density (bool): if true, the points are sorted, by LOF density, for order of importance in gravity.
                This parameter does not have any impact in Newtonian method because the overall movement in the Cartesian coordinate system is equivalent to summation os movements. 
            method (str): the method for GDR algorithm, i.e., Newtonian and Relativity. 
                Default is the Newtonian method. 
            metric (str): the metric used in the Relativity method. 
                Options are Schwarzschild (for general relativity) and Minkowski (for special relativity). Schwarzschild works much better.
                This is only used for the Relativity method (and not for the Newtonian method).
            use_PCA_for_Newtonian (bool): whether to use PCA for the Newtonian method. 
                If False, then the Newtonian movement in done in the input space, rather than the 3D PCA space.
                This variable is only important for the Newtonian method and not the Relativity method. PCA is always applied for the Relativity method.
        """
        self._max_itrations = max_itrations
        self._alpha = alpha
        self._alpha = self._alpha / np.sum(self._alpha)  #--> make sure they sum to one
        self._supervised_mode = supervised_mode
        self._do_sort_by_density = do_sort_by_density
        self._method = method
        self._metric = metric
        self._use_PCA_for_Newtonian = use_PCA_for_Newtonian
        self._n_samples = None
        self._dimensionality = None
        self._n_classes = None
        self._class_names = None

    def fit_transform(self, D: np.ndarray, labels: Optional[np.array] = None) -> np.ndarray:
        """
        Fit and transform the data for dimensionality reduction.

        Args:
            D (np.ndarray): The row-wise dataset, with rows as samples and columns as features
            labels (np.array): The labels of samples, if the samples are labeled. 

        Returns:
            D_transformed (np.ndarray): The row-wise transformed dataset, with rows as samples and columns as features
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
        if (self._method == 'Newtonian') and (not self._use_PCA_for_Newtonian):
            X = D.copy()
        else:
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

        if SHOW_VISUALIZATION or SAVE_VISUALIZATION: 
            X_final, labels_final = self._unsort_and_convertToX_if_necessary(X=X, X_classes=X_classes, labels=labels, sorted_indices=sorted_indices, indices_classes=indices_classes)
            plt1 = utils.plot_3D(X=X_final.T, labels=labels_final, class_names=self._class_names)
            if SHOW_VISUALIZATION: plt1.show()
            if SAVE_VISUALIZATION: 
                if not os.path.exists(PATH_SAVE): os.makedirs(PATH_SAVE)
                plt1.savefig(PATH_SAVE+'3D_before_iterations.png')
                if not os.path.exists(PATH_SAVE+'plot_files/'): os.makedirs(PATH_SAVE+'plot_files/')
                utils.save_variable(variable=X_final, name_of_variable='before_iterations_X', path_to_save=PATH_SAVE+'plot_files/')
                utils.save_variable(variable=labels_final, name_of_variable='before_iterations_labels', path_to_save=PATH_SAVE+'plot_files/')
            if (self._method == 'Newtonian') and (not self._use_PCA_for_Newtonian):
                D_transformed = X_final.T
            else:
                D_transformed = pca.inverse_transform(X=X_final.T)
            plt2 = utils.plot_embedding_of_points(embedding=D_transformed, labels=labels_final, class_names=self._class_names, n_samples_plot=2000, method='tsne')
            if SHOW_VISUALIZATION: plt2.show()
            if SAVE_VISUALIZATION: 
                plt2.savefig(PATH_SAVE+f'highDim_before_iterations.png')
                utils.save_variable(variable=D_transformed, name_of_variable=f'before_iterations_D', path_to_save=PATH_SAVE+'plot_files/')
            plt1.close()
            plt2.close()

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
            if SHOW_VISUALIZATION or SAVE_VISUALIZATION: 
                X_final, labels_final = self._unsort_and_convertToX_if_necessary(X=X, X_classes=X_classes, labels=labels, sorted_indices=sorted_indices, indices_classes=indices_classes)
                plt1 = utils.plot_3D(X=X_final.T, labels=labels_final, class_names=self._class_names)
                if SHOW_VISUALIZATION: plt1.show()
                if SAVE_VISUALIZATION: 
                    if not os.path.exists(PATH_SAVE): os.makedirs(PATH_SAVE)
                    plt1.savefig(PATH_SAVE+f'3D_itr_{itr}.png')
                    if not os.path.exists(PATH_SAVE+'plot_files/'): os.makedirs(PATH_SAVE+'plot_files/')
                    utils.save_variable(variable=X_final, name_of_variable=f'itr_{itr}_X', path_to_save=PATH_SAVE+'plot_files/')
                    utils.save_variable(variable=labels_final, name_of_variable=f'itr_{itr}_labels', path_to_save=PATH_SAVE+'plot_files/')
                if (self._method == 'Newtonian') and (not self._use_PCA_for_Newtonian):
                    D_transformed = X_final.T
                else:
                    D_transformed = pca.inverse_transform(X=X_final.T)
                plt2 = utils.plot_embedding_of_points(embedding=D_transformed, labels=labels_final, class_names=self._class_names, n_samples_plot=2000, method='tsne')
                if SHOW_VISUALIZATION: plt2.show()
                if SAVE_VISUALIZATION: 
                    plt2.savefig(PATH_SAVE+f'highDim_itr_{itr}.png')
                    utils.save_variable(variable=D_transformed, name_of_variable=f'itr_{itr}_D', path_to_save=PATH_SAVE+'plot_files/')
                plt1.close()
                plt2.close()

        # Unsort and convert X_classes to X, if necessary:
        X_final, labels_final = self._unsort_and_convertToX_if_necessary(X=X, X_classes=X_classes, labels=labels, sorted_indices=sorted_indices, indices_classes=indices_classes)

        # reconstruct from PCA subspace (space manifold in physics):
        if (self._method == 'Newtonian') and (not self._use_PCA_for_Newtonian):
            D_transformed = X_final.T
        else:
            D_transformed = pca.inverse_transform(X=X_final.T)  # NOTE: D_transformed is row-wise
        

        return D_transformed

    def _main_algorithm_Newtonian(self, X: np.ndarray) -> np.ndarray:
        """
        The main GDR algorithm for the Newtonian method. 

        Args:
            X (np.ndarray): The column-wise dataset, with columns as samples and rows as features

        Returns:
            X (np.ndarray): The column-wise transformed dataset, with columns as samples and rows as features.
        """
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

    def _main_algorithm_Relativity(self, X: np.ndarray) -> np.ndarray:
        """
        The main GDR algorithm for the Relativity method. 

        Args:
            X (np.ndarray): The column-wise dataset, with columns as samples and rows as features

        Returns:
            X (np.ndarray): The column-wise transformed dataset, with columns as samples and rows as features.
        """
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
                assert (np.sum(self._alpha) == 1)
                # amount of movement:
                movement_amount = 1/r_ij
                movement_amount_in_r = movement_amount * self._alpha[0]
                movement_amount_in_theta = movement_amount * self._alpha[1]
                movement_amount_in_phi = movement_amount * self._alpha[2]
                # calculate r and theta:
                r, theta = self._caculate_r_and_theta(origin=x_i, x=x_j)
                if DEBUG_MODE and VERBOSITY >= 3: print('r, theta, x_i, x_j: ', r, theta, x_i, x_j)
                # tensor components:
                if self._metric == "Schwarzschild":
                    g = self._Schwarzschild_metric(r=r, theta=theta, M=1, G=1, c=1, ignore_time_component=True)
                elif self._metric == "Minkowski":
                    g = self._Minkowski_metric(c=1, ignore_time_component=True)
                r_component = g[0, 0]
                theta_component = g[1, 1]
                phi_component = g[2, 2]
                if DEBUG_MODE and VERBOSITY >= 3: print('r_component, theta_component, phi_component: ', r_component, theta_component, phi_component)
                # movement in r, theta, and phi directions:
                if r_component > 0:
                    delta_ij_value_r = -1 * (movement_amount_in_r / r_component)**0.5
                else:
                    delta_ij_value_r = 0
                delta_ij_value_theta = -1 * (movement_amount_in_theta / theta_component)**0.5
                delta_ij_value_phi = -1 * (movement_amount_in_phi / phi_component)**0.5
                delta_ij = [delta_ij_value_r, delta_ij_value_theta, delta_ij_value_phi]
                # overall movement:
                if self._metric == "Schwarzschild":
                    if DEBUG_MODE and VERBOSITY >= 3: print('delta_ij: ', delta_ij)
                    x_j = self._move_in_spherical_coordinate_system(x=x_j, origin=x_i, delta=delta_ij)
                elif self._metric == "Minkowski":
                    x_j = x_j + delta_ij
            X[:, -j] = x_j
        return X

    def _move_in_spherical_coordinate_system(self, x: np.array, origin: np.array, delta: np.array) -> np.array:
        """
        Move a point in the spherical coordinate system.  

        Args:
            x (np.array): the vector in the spherical coordinate system, in the format of (r, theta, phi)
            origin (np.array): the origin vector in the spherical coordinate system, in the format of (r, theta, phi)
            delta (np.array): the movement (translation) vector in the spherical coordinate system, in the format of (delta_r, delta_theta, delta_phi)

        Returns:
            x (np.array): the moved (translated) vector in the spherical coordinate system, in the format of (r, theta, phi)
        """
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

    def _convert_Cartesian_to_spherical_coordinates(self, x: np.array) -> np.array:
        """
        Convert the point coordinates in Cartesian coordinate system to the point coordinates in the spherical coordinate system.

        Args:
            x (np.array): the vector in the Cartesian coordinate system, in the format of (x, y, z)

        Returns:
            x (np.array): the vector in the spherical coordinate system, in the format of (r, theta, phi)

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

    def _convert_spherical_to_Cartesian_coordinates(self, x: np.array) -> np.array:
        """
        Convert the point coordinates in spherical coordinate system to the point coordinates in Cartesian coordinate system.

        Args:
            x (np.array): the vector in the spherical coordinate system, in the format of (r, theta, phi)

        Returns:
            x (np.array): the vector in the Cartesian coordinate system, in the format of (x, y, z)

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

    def _caculate_r_and_theta(self, origin: np.array, x: np.array) -> Tuple[float, float]:
        """
        Calculate r and theta in the spherical coordinate system with a specified origin.

        Args:
            origin (np.array): the origin vector in the spherical coordinate system, in the format of (r, theta, phi)
            x (np.array): the vector in the spherical coordinate system, in the format of (r, theta, phi)

        Returns:
            r (float): the r component of vector in the spherical coordinate system
            theta (float): the theta component of vector in the spherical coordinate system

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

    def _Schwarzschild_metric(self, r: float, theta: float, M: Optional[float] = 1, G: Optional[float] = 1, 
                            c: Optional[float] = 1, ignore_time_component: Optional[bool] = False) -> np.ndarray:
        """
        Calculate the Schwarzschild metric in the general relativity.

        Args:
            r (float): the r component of vector in the spherical coordinate system
            theta (float): the theta component of vector in the spherical coordinate system
            M (float): the mass of gravitational particle
            G (float): the gravitational constant
            c (float): the speed of light
            ignore_time_component (bool): whether to ignore the time component of metric.
                if true, metric is 3*3; otherwise, metric is 4*4

        Returns:
            metric (np.ndarray): the metric of general relativity as a 3*3 or 4*4 matrix

        Notes:
            https://en.wikipedia.org/wiki/Metric_tensor_(general_relativity)
        """
        temp = 1 - ((2 * G * M) / (r * (c**2)))
        if not ignore_time_component:
            t_component = -1 * temp
        r_component = (1 / temp)
        theta_component = r**2
        phi_component = (r**2) * (math.sin(theta)**2)
        if not ignore_time_component:
            metric = np.diag([t_component, r_component, theta_component, phi_component])
        else:
            metric = np.diag([r_component, theta_component, phi_component])
        return metric

    def _Minkowski_metric(self, c: Optional[float] = 1, ignore_time_component: Optional[bool] = False) -> np.ndarray:
        """
        Calculate the Minkowski metric in the general relativity.

        Args:
            c (float): the speed of light
            ignore_time_component (bool): whether to ignore the time component of metric.
                if true, metric is 3*3; otherwise, metric is 4*4

        Returns:
            metric (np.ndarray): the metric of general relativity as a 3*3 or 4*4 matrix

        Notes:
            https://en.wikipedia.org/wiki/Metric_tensor_(general_relativity)
        """
        if not ignore_time_component:
            t_component = -1 * (c**2)
        r_component = 1
        theta_component = 1
        phi_component = 1
        if not ignore_time_component:
            metric = np.diag([t_component, r_component, theta_component, phi_component])
        else:
            metric = np.diag([r_component, theta_component, phi_component])
        return metric

    def _sort_by_density(self, X: np.ndarray, labels: Optional[np.array] = None) -> Tuple[np.ndarray, Optional[np.array], List[int]]:
        """
        Sort the samples based on density of Local Outlier Factor (LOF).

        Args:
            X (np.ndarray): The column-wise dataset, with columns as samples and rows as features
            labels (np.array): The labels of samples, if the samples are labeled

        Returns:
            X (np.ndarray): The sorted column-wise dataset, with columns as samples and rows as features
            labels (np.array): The sorted labels of samples, if the samples are labeled
            sorted_indices (List[int]): the sorted indices

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

    def _unsort(self, sorted_indices: List[int], X: np.ndarray, labels: Optional[np.array] = None) -> Tuple[np.ndarray, Optional[np.array]]:
        """
        Unsort the samples based on the sorted indices.

        Args:
            sorted_indices (List[int]): the sorted indices
            X (np.ndarray): The sorted column-wise dataset, with columns as samples and rows as features
            labels (np.array): The sorted labels of samples, if the samples are labeled

        Returns:
            X_unsorted (np.ndarray): The unsorted column-wise dataset, with columns as samples and rows as features
            labels_unsorted (np.array): The unsorted labels of samples, if the samples are labeled
        """
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

    def _convert_X_to_classes(self, X: np.ndarray, labels: np.array) -> Tuple[List[np.ndarray], List[np.array]]:
        """
        Convert (separate) X to classes.

        Args:
            X (np.ndarray): the column-wise dataset, with columns as samples and rows as features
            labels (np.array): the labels of samples, if the samples are labeled

        Returns:
            X_classes (List[np.ndarray]): the list of data points inside each class
                The matrix of every class is column-wise, with columns as samples and rows as features.
            indices_classes (List[np.array]): the indices of points for every class
        """
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

    def _convert_classes_to_X(self, X_classes: List[np.ndarray], indices_classes: List[np.array]) -> np.ndarray:
        """
        Convert (accumulate) classes to dataset.

        Args:
            X_classes (List[np.ndarray]): the list of data points inside each class. 
                The matrix of every class is column-wise, with columns as samples and rows as features.
            indices_classes (List[np.array]): the indices of points for every class

        Returns:
            X (np.ndarray): the column-wise dataset, with columns as samples and rows as features 
        """
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

    def _unsort_and_convertToX_if_necessary(self, X: np.ndarray, X_classes: List[np.ndarray], labels: Union[np.array, None], 
                            sorted_indices: List[int], indices_classes: List[np.array]) -> Tuple[np.ndarray, np.array]:
        """
        Unsort and convert X_classes to X, if necessary.

        Args:
            X (np.ndarray): the column-wise dataset, with columns as samples and rows as features 
            X_classes (List[np.ndarray]): the list of data points inside each class. 
                The matrix of every class is column-wise, with columns as samples and rows as features.
            labels (np.array): the labels of samples, if the samples are labeled
            sorted_indices (List[int]): the sorted indices
            indices_classes (List[np.array]): the indices of points for every class

        Returns:
            X_final (np.ndarray): the data of plot.
            labels_final (np.array): the labels of plot (for color of plot).
        """
        if not self._supervised_mode:
            if self._do_sort_by_density:
                X_final, labels_final = self._unsort(sorted_indices, X, labels)
            else:
                X_final, labels_final = X.copy(), labels.copy()
        else:
            if self._do_sort_by_density:
                X_classes_unsorted = X_classes.copy()
                for label in range(self._n_classes):
                    X_classes_unsorted[label] = self._unsort(sorted_indices=sorted_indices[label], X=X_classes[label])
                X_final = self._convert_classes_to_X(X_classes_unsorted, indices_classes)
            else:
                X_final = self._convert_classes_to_X(X_classes, indices_classes)
            labels_final = labels.copy()
        return X_final, labels_final

    def test_Newtonian_movement(self) -> None:
        """Test Newtonian movement for two test points."""
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

    def test_Relativity_movement(self) -> None:
        """Test Relativity movement for two test points."""
        x_i = np.array([1, 1, 1])
        x_j = np.array([2, 3, 4])
        r_ij = np.linalg.norm(x_i - x_j)
        # weights:
        assert (np.sum(self._alpha) == 1)
        # amount of movement:
        movement_amount = 1/r_ij
        movement_amount_in_r = movement_amount * self._alpha[0]
        movement_amount_in_theta = movement_amount * self._alpha[1]
        movement_amount_in_phi = movement_amount * self._alpha[2]
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