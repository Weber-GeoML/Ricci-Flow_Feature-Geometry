from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import numpy as np
from scipy.stats import pearsonr
from ricci_curvatures import Ricci_Curvature_Calculator
import warnings


class Ricci_Coefficients:

    def __init__(self, NN, X, k):
        """
        Initialize the Ricci_Coefficients calculator

        Args:
            NN (torch.nn.Module): PyTorch neural network.
            X (torch.Tensor): Point cloud.
            k (int):  Number of neighbors in k-nearest neighbor graph.
        """
        self.NN = NN
        self.X = X
        self.k = k

    def global_ricci_coefficient(self):
        """Calculate the global Ricci coefficient introduced by Baptista et al. in "Deep Learning as Ricci Flow" (2024)

        Raises:
            ValueError: If one of the k-nearest-neighbor graphs fails to be connected.

        Returns:
            float: Ricci coefficient. 
        """
        # 1. Construct k-nearest-neighbor graphs from point clouds
        features = self.NN.features(self.X)
        kNN_graphs = []
        for feature_samples in features:
            # Directed k-nearest-neighbor graph
            graph_directed = kneighbors_graph(feature_samples, self.k, mode='connectivity', include_self=False)
            # Undirected k-nearest-neighbor graph
            graph_undirected = graph_directed.maximum(graph_directed.T)
            kNN_graphs.append(graph_undirected)
        
        # 2. Calculate shortest-path-distances
        apsps = []
        for graph in kNN_graphs:
            apsps.append(dijkstra(csgraph=graph, directed=False, unweighted=True, return_predecessors=False))

        # 3. Calculate Forman Ricci curvature
        forman_curvatures = []
        for graph in kNN_graphs:
            forman_curvatures.append(Ricci_Curvature_Calculator(A=graph).forman_ricci().sum() / 2)
        
        # 4. Calculate expansion/contraction
        g = []
        for apsp in apsps:
            g.append(apsp.sum()/2)
        
        # 5. Check that all k-nearest-neighbor graphs are connected
        if np.any(np.isinf(g)):
            raise ValueError('Constructed k-Nearest-Neighbour Graph is not connected.')
        
        # 6. Calculate Pearson correlation coefficient
        eta = []
        for l in range(len(g) - 1):
            eta.append(g[l+1] - g[l])
        ricci_coefficient, _ = pearsonr(forman_curvatures[:-1], eta)
        return ricci_coefficient

    def local_ricci_coefficient(self, curv):
        """
        Calculate the local Ricci coefficients.

        Args:
            curv (str): Type of discrete Ricci curvature.

        Raises:
            ValueError: Discrete Ricci curvature not supported
            ValueError: One-hop nieghborhood is not connected in subsequent layer.

        Returns:
            np.ndarray: A 1D array of floats containing the local Ricci coefficients.
        """

        # 1. Construct k-nearest-neighbor graphs from point clouds
        features = self.NN.features(self.X)
        depth = len(features)
        kNN_graphs = []
        for feature_samples in features:
            # Directed k-nearest-neighbor graph
            graph_directed = kneighbors_graph(feature_samples, self.k, mode='connectivity', include_self=False)
            # Undirected k-nearest-neighbor graph
            graph_undirected = graph_directed.maximum(graph_directed.T)
            kNN_graphs.append(graph_undirected)

        # 2. Calculate shortest-path-distances
        apsps = []
        for graph in kNN_graphs:
            apsps.append(dijkstra(csgraph=graph, directed=False, unweighted=True, return_predecessors=False))

        # 3. Calculate the Ricci curvature of every k-nearest-neighbour graph
        curvatures = []
        for i, graph in enumerate(kNN_graphs[:-1]):
            if curv == 'Forman-Ricci':
                curvatures.append(Ricci_Curvature_Calculator(A=graph).forman_ricci())

            elif curv == 'Augmented-Forman-Ricci':  
                curvatures.append(Ricci_Curvature_Calculator(A=graph).augmented_forman_ricci())

            elif curv == 'Approx-Ollivier-Ricci':
                curvatures.append(Ricci_Curvature_Calculator(A=graph).approx_ollivier_ricci())

            elif curv == 'Ollivier-Ricci':
                curvatures.append(Ricci_Curvature_Calculator(A=graph).ollivier_ricci(apsp=apsps[i]))
                print(f"Curvature of layer {i} calculated", end="\r")
            else: 
                raise ValueError("Ricci curvature notion not supported.")
            
        # 4. Calculate the local Ricci coefficients
        local_ricci_coefficients = np.empty(len(features[0]))
        for x in range(len(features[0])):
            # 4.1 Calculate scalar curvature
            scalar_curvs = []
            for i in range(depth-1):
                scalar_curvs.append(np.divide(curvatures[i][x].sum(), kNN_graphs[i][x].count_nonzero()))

            # 4.2 Calculate local expansion/contraction
            one_hop_neighborhoods_connected = True
            eta = []
            for i in range(depth-1):
                S1 = kNN_graphs[i][x].indices   
                ec = 0
                for y in S1:
                    if apsps[i+1][x,y] == np.inf:
                        warnings.warn('One hop-neighbors are not connected in the subsequent layer.')
                        one_hop_neighborhoods_connected = False
                    else:
                        ec += apsps[i+1][x,y] - apsps[i][x,y]
                eta.append(ec/len(S1)) 
            if one_hop_neighborhoods_connected:
                lrc, _ = pearsonr(scalar_curvs, eta)
                local_ricci_coefficients[x] = lrc
            else: 
                local_ricci_coefficients[x] = np.nan

        return local_ricci_coefficients
    
    def layer_ricci_coefficient(self, curv):
        """
        Calculate the layer Ricci coefficients.

        Args:
            curv (str): Type of discrete Ricci curvature.

        Raises:
            ValueError: Discrete Ricci curvature not supported.
            ValueError: One-hop nieghborhood is not connected in subsequent layer.

        Returns:
            np.ndarray: A 1D array of floats containing the layer Ricci coefficients.
        """
        # 1. Construct the k-nearest-neighbor graphs
        features = self.NN.features(self.X)
        depth = len(features)
        kNN_graphs = []
        for feature_samples in features:
            # Directed k-nearest-neighbor graph
            graph_directed = kneighbors_graph(feature_samples, self.k, mode='connectivity', include_self=False)
            # Undirected k-nearest-neighbor graph
            graph_undirected = graph_directed.maximum(graph_directed.T)
            kNN_graphs.append(graph_undirected)
        
        # 2. Calculate shortest-path-distances
        apsps = []
        for graph in kNN_graphs:
            apsps.append(dijkstra(csgraph=graph, directed=False, unweighted=True, return_predecessors=False))

        # 3. Calculate the Ricci curvature of every k-nearest-neighbour graph
        curvatures = []
        for i, graph in enumerate(kNN_graphs[:-1]):
            if curv == 'Forman-Ricci':
                curvatures.append(Ricci_Curvature_Calculator(A=graph).forman_ricci())

            elif curv == 'Augmented-Forman-Ricci':  
                curvatures.append(Ricci_Curvature_Calculator(A=graph).augmented_forman_ricci())

            elif curv == 'Approx-Ollivier-Ricci':
                curvatures.append(Ricci_Curvature_Calculator(A=graph).approx_ollivier_ricci())

            elif curv == 'Ollivier-Ricci':
                curvatures.append(Ricci_Curvature_Calculator(A=graph).ollivier_ricci(apsp=apsps[i]))
                print(f"Curvature of layer {i} calculated", end="\r")
            else: 
                raise ValueError("Ricci curvature notion not supported.")
        
        # 4. Calculate the Ricci coefficient of every layer
        layer_ricci_coefficients = np.empty(depth-1)
        for i in range(depth - 1):
            scalar_curvs = []
            eta = [] 
            for x in range(len(features[0])):
                S1 = kNN_graphs[i][x].indices 
                ec = 0
                one_hop_neighborhoods_connected = True
                # 4.1 Calculate expansion/contraction
                for y in S1:
                    if apsps[i+1][x,y] == np.inf:
                        warnings.warn('One hop-neighbors are not connected in the subsequent layer.')
                        one_hop_neighborhoods_connected = False
                    else:
                        ec += apsps[i+1][x,y] - apsps[i][x,y]
                # 4.2 Calculate scalar curvature
                if one_hop_neighborhoods_connected:
                    scalar_curvs.append(np.divide(curvatures[i][x].sum(), kNN_graphs[i][x].count_nonzero()))
                    eta.append(ec/len(S1))  
            lrc, _ = pearsonr(scalar_curvs, eta)
            layer_ricci_coefficients[i] =lrc 

        return layer_ricci_coefficients
