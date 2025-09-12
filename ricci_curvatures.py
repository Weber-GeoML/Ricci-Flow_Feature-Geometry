import numpy as np
import ot
import multiprocessing as mp
from scipy.sparse import triu, csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra


class Ricci_Curvature_Calculator:
    """
    Class to calculate discrete Ricci curvatures of a graph. 

    The class accepts an adjacency matrix of the graph and provides methods to compute:
      - Forman Ricci curvature
      - Augmented Forman Ricci curvature
      - (Approximate) Ollivier Ricci curvature
      - Ollivier Ricci curvature
    """

    def __init__(self, A):
        """
        Initialize the calculator.

        Args:
            A (csr_matrix): A 2D sparse adjacency matrix in Compressed Sparse Row (CSR) format
        """
        self.A = csr_matrix(A)
    
    def forman_ricci(self):
        """
        Compute the augmented Forman Ricci curvature of the graph.

        Returns:
            np.ndarray: Symmetric matrix of shape (n_nodes, n_nodes) containing the augmented Forman Ricci curvature for each edge. Non-edge entries remain 0.
        """
        degrees = self.A @ np.ones(self.A.shape[0])
        Ric = lil_matrix(self.A.shape, dtype=np.int32)  # use LIL for assignment
        A_upper = triu(self.A, k=1)
        rows, cols = A_upper.nonzero()
        for i, j in zip(rows, cols):  
            Ric[i,j] = 4 - degrees[i] - degrees[j]
            Ric[j,i] = Ric[i,j]
        return Ric.tocsr()
    
    def augmented_forman_ricci(self):
        """
        Compute the augmented Forman Ricci curvature.

        Returns:
            np.ndarray: Symmetric matrix of shape (n_nodes, n_nodes) containing the augmented Forman Ricci curvature for each edge. Non-edge entries remain 0.
        """
        degrees = self.A @ np.ones(self.A.shape[0])
        A2 = self.A @ self.A
        Ric = lil_matrix(self.A.shape, dtype=np.int32)  # use LIL for assignment
        A_upper = triu(self.A, k=1)
        rows, cols = A_upper.nonzero()
        for i, j in zip(rows, cols):  
                trianlges = A2[i,j]
                Ric[i,j] = 4 - degrees[i] - degrees[j] + 3 * trianlges
                Ric[j,i] = Ric[i,j]
        return Ric.tocsr()
    
    def approx_ollivier_ricci(self):
        """
        Compute the Approximation of the Ollivier-Ricci curvature proposed in Tian et al. "Curvature-based Clustering on Graphs" (2023)

        Returns:
            np.ndarray: Symmetric matrix of shape (n_nodes, n_nodes) containing the approximation for each edge. Non-edge entries remain 0.
        """
        degrees = self.A @ np.ones(self.A.shape[0])
        A2 = self.A @ self.A
        Ric = lil_matrix(self.A.shape, dtype=np.float32)  # use LIL for assignment
        A_upper = triu(self.A, k=1)
        rows, cols = A_upper.nonzero()
        for i, j in zip(rows, cols):
            triangles = A2[i,j]
            Ric[i,j] = 1/2 * (triangles / max(degrees[i], degrees[j])) - 1/2 * (max(0, 1- 1/degrees[i] - 1/degrees[j] - triangles / min(degrees[i], degrees[j])) + max(0, 1- 1/degrees[i] - 1/degrees[j] - triangles / max(degrees[i], degrees[j])) - triangles/max(degrees[i], degrees[j]))
            Ric[j,i] = Ric[i,j]
        return Ric.tocsr()

    def _orc_edge(self, e_apsp):
        """Private helper: Ollivier-Ricci curvature for one edge (used in multiprocessing)."""
        e, apsp = e_apsp  
        S_1x = self.A[e[0]].indices
        S_1y = self.A[e[1]].indices
        d_x = len(S_1x)
        d_y = len(S_1y)
        cost_matrix = np.array([[apsp[l, k] for l in S_1y] for k in S_1x])
        W = ot.emd2([1/d_x]*d_x, [1/d_y]*d_y, cost_matrix)
        return [e, 1 - W]

    def ollivier_ricci(self, apsp=None):
        """
        Compute the Ollivier-Ricci curvature of all edges in the graph using multiprocessing. It requires the all-pairs shortest path (APSP) matrix, which must be precomputed.

        Args:
            apsp (np.ndarray): Precomputed all-pairs shortest path matrix of shape (n_nodes, n_nodes).

        Returns:
            np.ndarray: Symmetric matrix of shape (n_nodes, n_nodes) containing the Ollivier-Ricci curvature for each edge. Non-edge entries remain 0.

        Raises:
            ValueError: If apsp is None.
    
        Notes:
            - This function uses multiprocessing for speed, so it can handle large graphs efficiently.
        """
        if apsp is None:
            apsp = dijkstra(csgraph=self.A, directed=False, unweighted=True,return_predecessors=False)
        
        A_upper = triu(self.A, k=1)
        edges = np.vstack(A_upper.nonzero()).T
        Ric = lil_matrix(self.A.shape, dtype=np.float32)  # use LIL for assignment

        args = [(tuple(e), apsp) for e in edges]
        with mp.get_context("fork").Pool(mp.cpu_count()) as pool:
            chunksize, extra = divmod(len(edges), mp.cpu_count() * 4)
            if extra:
                chunksize += 1
            result = pool.imap_unordered(self._orc_edge, args, chunksize=chunksize)
            for v in result:
                i, j = v[0]
                Ric[i, j] = v[1]
                Ric[j, i] = v[1]

        return Ric.tocsr()
    
