from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
import numpy as np
import networkx as nx
from ricci_curvatures import Ricci_Curvature_Calculator



class CommunityStructure:
    """
    Class to evaluate the emergence of community structure on the k-nearest neighbor graphs measured by curvature gaps, modularity, normalized cut,...
    """
    def __init__(self, NN):
        """
        Initialize the Community Structure class

        Args:
            NN (torch.nn.Module): PyTorch neural network.
        """
        self.NN = NN

    def ricci_curvature_distribution(self, X, y, curv, k):
        """
        Compute the distributions of Ricci curvatures for edges within the same class and between different classes in k-nearest neighbor graphs of neural network features.

        Args:
            X (torch.Tensor): Input data
            y (torch.Tensor): True labels
            curv (str): Ricci curvature notion to use.
            k (int):  Number of neighbors in k-nearest neighbor graph.

        Raises:
            ValueError: Curvature notion not supported.

        Returns:
            List[np.ndarray]: List containing arrays of Ricci curvature values for edges connecting nodes with the same label in each feature set.
            List[np.ndarray]: List containing arrays of Ricci curvature values for edges connecting nodes with different labels in each feature set.
        """
        features = self.NN.features(X)
        within_curvature_distributions = []
        between_curvature_distributions = []

        for feature_samples in features:
            # Directed k-nearest-neighbor graph
            graph_directed = kneighbors_graph(feature_samples, k, mode='connectivity', include_self=False)
            # Undirected k-nearest-neighbor graph
            A = graph_directed.maximum(graph_directed.T)

            # Calculate Ricci curvature
            calculator = Ricci_Curvature_Calculator(A=A)
            if curv == "Ollivier-Ricci":
                apsp = dijkstra(csgraph=A, directed=False, unweighted=True,return_predecessors=False)
                ricci_curvatures = calculator.ollivier_ricci(apsp=apsp)
            elif curv == "Approx-Ollivier-Ricci":
                ricci_curvatures = calculator.approx_ollivier_ricci()
            elif curv == "Augmented-Forman-Ricci":
                ricci_curvatures = calculator.augmented_forman_ricci()
            else:
                raise ValueError("Ricci curvature notion not supported.")
            
            within_curvatures = []
            between_curvatures = []
            for i, j in zip(*A.nonzero()):
                if i < j:
                    if y[i] == y[j]:
                        within_curvatures.append(ricci_curvatures[i, j])
                    else:
                        between_curvatures.append(ricci_curvatures[i, j])

            within_curvature_distributions.append(within_curvatures)
            between_curvature_distributions.append(between_curvatures)
        
        return within_curvature_distributions, between_curvature_distributions
    
    def curvature_gap(self, X, y, curv, k):
        """
        Compute the curvature gap between within-class and between-class edges in k-nearest neighbor graphs of neural network features.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): True labels.
            curv (str): Ricci curvature notion to use.
            k (int):  Number of neighbors in k-nearest neighbor graph.

        Returns:
            List[float]: Curvature gaps
        """
        within_curvature_distributions, between_curvature_distributions = self.ricci_curvature_distribution(X=X, y=y, curv=curv, k=k)
        # Compute mean and std for within- and between-class edges
        kappa_within = [np.mean(c) for c in within_curvature_distributions]
        sigma_within = [np.std(c) for c in within_curvature_distributions]

        kappa_between = [np.mean(c) for c in between_curvature_distributions]
        sigma_between = [np.std(c) for c in between_curvature_distributions]

        # Compute curvature gap
        curvature_gap = [
            (kappa_within[i] - kappa_between[i]) / np.sqrt(0.5 * (sigma_within[i]**2 + sigma_between[i]**2))
            for i in range(len(kappa_within))
        ]

        return curvature_gap
    
    def modularity(self, X, y, k):
        """
        Compute the modularity of the true class communities on k-nearest neighbor graphs of the neural network feature representations for each layer.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): True labels.
            k (int): Number of neighbors in k-nearest neighbor graph.

        Returns:
            List[float]: modularity of each layer
        """
        modularities = []
        features = self.NN.features(X)
        communities = [[i for i, val in enumerate(y) if val == label] for label in [0,1]]
        for feature_samples in features:
            # Directed k-Nearest-Neighbour Graph
            graph_directed = kneighbors_graph(feature_samples, k, mode='connectivity', include_self=False)
            # Undirected k-Nearest-Neighbour Graph
            A = graph_directed.maximum(graph_directed.T)
            # Convert to NetworkX graph
            G = nx.from_scipy_sparse_array(A)

            modularities.append(nx.community.modularity(G, communities))

        return modularities
    
    def normalized_cut(self, X, y, k):
        """
        Compute the normalized cut of the true class communities on k-nearest neighbor graphs of the neural network feature representations for each layer.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): True labels.
            k (int): Number of neighbors in the k-nearest-neighbor graph.

        Returns:
            List[float]: Normalized cut of each layer.
        """
        normalized_cuts = []
        features = self.NN.features(X)
        communities = [[i for i, val in enumerate(y) if val == label] for label in [0,1]]

        for feature_samples in features:
            # Directed k-Nearest-Neighbour Graph
            graph_directed = kneighbors_graph(feature_samples, k, mode='connectivity', include_self=False)
            # Undirected k-Nearest-Neighbour Graph
            A = graph_directed.maximum(graph_directed.T)
            # Convert to NetworkX graph
            G = nx.from_scipy_sparse_array(A)

            n_cut = 0.5 * nx.algorithms.cuts.normalized_cut_size(G=G, S=communities[0], T=communities[1])
            normalized_cuts.append(n_cut)
        return normalized_cuts