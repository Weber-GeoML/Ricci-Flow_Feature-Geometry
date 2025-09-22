import torch 
from torchvision import datasets, transforms
import numpy as np
from sklearn.datasets import make_swiss_roll, make_circles


class DatasetFactory:
    """
    Factory class to generate datasets for experiments.
    """
    @staticmethod
    def make_circles(n=1000, noise=0.05):
        """Generate a 2D dataset of points forming concentric circles.

        Args:
            n (int, optional): Number of samples. Defaults to 1000.
            noise (float, optional): Standard deviation of Gaussian noise added to the data. Defaults to 0.05.

        Returns:
            X (torch.Tensor): Tensor of shape (n, 2) containing the coordinates of the points.
            y (torch.Tensor): Tensor of shape (n,) containing class labels (0 or 1) as integers.
        """
        X,y = make_circles(n_samples=n, noise=noise)

        return torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.float)
    
    @staticmethod
    def make_4circles(n=1000, r=0.2):
        """Generate points from the unit circle with labels depending on whether they lie inside one of four smaller inner circles.

        Args:
            n (int, optional): Number of samples. Defaults to 1000.
            r (float, optional): Radius of the smaller circles. Defaults to 0.2.


        Returns:
            X (torch.Tensor): Tensor of shape (n, 2) containing the coordinates of the points.
            y (torch.Tensor): Tensor of shape (n,) containing class labels (0 or 1) as integers.
        """
        radii = np.sqrt(np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)
        x1 = radii * np.cos(theta)
        x2 = radii * np.sin(theta)
        X = np.stack((x1, x2), axis=1)

        centers = np.array([[0, 0.5], [-0.5, 0], [0, -0.5], [0.5, 0]])
        y = np.zeros(n, dtype=np.int64)

        for cx, cy in centers:
            mask = (X[:, 0] - cx)**2 + (X[:, 1] - cy)**2 <= r**2
            y[mask] = 1

        return torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.float)
    
    @staticmethod
    def make_cylinders(r1=1, r2=0.8, n=1000, noise=0.1):
        """Generate a 3D dataset of points arranged in two concentric cylindrical shells along XZ-plane

        Args:
            r1 (float, optional): Radius of the first cylinder (class 0). Defaults to 1.
            r2 (float, optional): Radius of the second cylinder (class 1). Defaults to 0.8.
            n (int, optional): Number of samples. Defaults to 1000.
            noise (float, optional): Amount of uniform noise added to the radii. Defaults to 0.1.

        Returns:
            X (torch.Tensor): Tensor of shape (n, 3) containing the 3D coordinates of the points.
            y (torch.Tensor): Tensor of shape (n,) containing class labels (0 or 1) as integers.
        """
        angles = np.random.uniform(0, 2 * np.pi, n)

        radii = np.ones(n)
        radii[:n//2] *= r1
        radii[n//2:] *= r2
        radii += np.random.uniform(-noise, noise, n)

        x1 = radii * np.cos(angles)
        x2 = np.random.rand(n)
        x3 = radii * np.sin(angles)
        X = np.stack((x1, x2, x3), axis=1).astype(np.float32)
        y = np.concatenate((np.zeros(n//2), np.ones(n-(n//2))))  

        return torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.float)
    
    @staticmethod
    def make_torus(R, r, u, v, center):
        """
        Helper function to generate 3D coordinates of points on a torus using its parametric equation.

        Args:
            R (float): Major radius.
            r (float or np.ndarray): Minor radius.
            u (np.ndarray): Angles around the central axis of torus.
            v (np.ndarray): Angles around the circular axis of the torus.
            center (tuple): 3D coordinates of center.

        Returns:
            x, y, z (np.ndarray): coordinates (same shape as u/v).
        """
        x = (R + r * np.cos(v)) * np.cos(u) + center[0]
        y = (R + r * np.cos(v)) * np.sin(u) + center[1]
        z = r * np.sin(v) + center[2]
        return x, y, z
    
    @staticmethod
    def make_tori(n=1000, R=2, r=1, center1=(0,0,0), center2=(2,0,0)):
        """
        Generate a 3D dataset consisting of two tori for binary classification.

        Args:
            n (int, optional): Number of samples. Defaults to 1000.
            R (float, optional): Major radius. Defaults to 2.
            r (float, optional): Minor radius. Defaults to 1.
            center1 (tuple, optional): Coordinates of the first torus center. Defaults to (0,0,0).
            center2 (tuple, optional): Coordinates of the second torus center. Defaults to (2,0,0).

        Returns:
            X (torch.Tensor): Tensor of shape (n, 3) containing 3D coordinates of all points.
            y (torch.Tensor): Tensor of shape (n,) with class labels (0 for torus 1, 1 for torus 2).
        """
        u1, v1 = np.random.uniform(0, 2 * np.pi, n // 2), np.random.uniform(0, 2 * np.pi, n // 2)
        u2, v2 = np.random.uniform(0, 2 * np.pi, n - n // 2), np.random.uniform(0, 2 * np.pi, n - n // 2)
        r1 = np.random.uniform(0,r, n // 2)
        r2 = np.random.uniform(0,r, n - n//2)
        x1, y1, z1 = DatasetFactory.make_torus(R=R, r=r1, u=u1, v=v1, center=center1)
        x2, z2, y2 = DatasetFactory.make_torus(R=R, r=r2, u=u2, v=v2, center=center2)

        X = np.concatenate((np.stack((x1, y1, z1), axis=1), np.stack((x2, y2, z2), axis=1)))
        y = np.concatenate((np.zeros(n//2), np.ones(n-n//2)))

        return torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.float)
    

    # -------------------------------- TorchVision Datasets --------------------------------
    @staticmethod
    def filter_digits(dataset, digits):
        """
        Keep only specified MNIST digits and remap to {0,1} with digits[1] -> 1.

        Args:
            dataset: MNIST-Dataset.
            digits (tuple): Digits we want to filter out.

        Returns:
            Filtered dataset
        """
        mask = torch.isin(dataset.targets, torch.tensor(digits))
        dataset.targets = dataset.targets[mask]
        dataset.data = dataset.data[mask]
        dataset.targets = (dataset.targets == digits[1]).long()
        return dataset

    
    @staticmethod
    def load_MNIST(digits, device='cpu'):
        """
        Function to load a binary dataset from the MNIST dataset.

        Args:
            digits (tuple): The two integers we want to consider.
            device (str): Device. Defaults to "cpu".

        Returns:
            tuple: Train and test datasets
        """
        train = datasets.MNIST(root="./data", train=True, download=True)
        test = datasets.MNIST(root="./data", train=False, download=True)

        train = DatasetFactory.filter_digits(train, digits=digits)
        test = DatasetFactory.filter_digits(test, digits=digits)

        X_train = (train.data.float() / 255.0).to(device)
        y_train = train.targets.float().to(device)

        # Balanced test subset (500 per class)
        idx0 = (test.targets == 0).nonzero(as_tuple=True)[0]
        idx1 = (test.targets == 1).nonzero(as_tuple=True)[0]
        k = min(500, len(idx0), len(idx1))
        sel = torch.cat((idx0[:k], idx1[:k]))
        X_test = (test.data[sel].float() / 255.0).to(device)
        y_test = test.targets[sel].float().to(device)

        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def filter_classes(dataset, classes):
        """
        Keep only specified Fashion-MNIST classes; remap to {0,1} with classes[1] -> 1.

        Args:
            dataset: Fashion-MNIST-Dataset.
            classes (tuple): Classes we want to filter out.

        Returns:
            Filtered dataset
        """
        mask = torch.isin(dataset.targets, torch.tensor(classes))
        dataset.targets = dataset.targets[mask]
        dataset.data = dataset.data[mask]
        dataset.targets = (dataset.targets == classes[1]).long()
        return dataset

    
    @staticmethod
    def load_fMNIST(classes, device="cpu"):
        """Function to load a binary dataset from the Fashion-MNIST dataset.

        Args:
            classes (tuple): The two classes we want to consider.
            device (str): Device. Defaults to "cpu".

        Returns:
            tuple: Train and test datasets.
        """
        train = datasets.FashionMNIST(root="./data", train=True, download=True)
        test = datasets.FashionMNIST(root="./data", train=False, download=True)

        train = DatasetFactory.filter_classes(train, classes=classes)
        test = DatasetFactory.filter_classes(test, classes=classes)

        X_train = (train.data.float() / 255.0).to(device)
        y_train = train.targets.float().to(device)

        # Balanced test subset (500 per class)
        idx0 = (test.targets == 0).nonzero(as_tuple=True)[0]
        idx1 = (test.targets == 1).nonzero(as_tuple=True)[0]
        k = min(500, len(idx0), len(idx1))
        sel = torch.cat((idx0[:k], idx1[:k]))
        X_test = (test.data[sel].float() / 255.0).to(device)
        y_test = test.targets[sel].float().to(device)

        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def filter_CIFAR(dataset, classes):
        """
        Keep only specified CIFAR classes; remap to {0,1} with classes[1] -> 1.

        Args:
            dataset: Fashion-MNIST-Dataset.
            classes (tuple): Classes we want to filter out.

        Returns:
            Filtered dataset
        """
        mask = torch.isin(torch.tensor(dataset.targets), torch.tensor(classes))
        dataset.targets = torch.tensor(dataset.targets)[mask]
        dataset.data = torch.tensor(dataset.data)[mask]
        dataset.targets = (dataset.targets == classes[1]).float()
        return dataset
    
    def load_CIFAR(classes, device="cpu"):
        """Function to load a binary dataset from the CIFAR-10 dataset.

        Args:
            classes (tuple): The two classes we want to consider.
            device (str): Device. Defaults to "cpu".

        Returns:
            tuple: Train and test datasets.
        """
        transform = transforms.ToTensor()
        # Load the full dataset
        full_train = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        full_test = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        # Filter classes
        train = DatasetFactory.filter_CIFAR(full_train, classes=classes)
        test = DatasetFactory.filter_CIFAR(full_test, classes=classes)

        X_train = (train.data.float() / 255.0).to(device)
        y_train = train.targets.float().to(device)

        # Balanced test subset (500 per class)
        idx0 = (test.targets == 0).nonzero(as_tuple=True)[0]
        idx1 = (test.targets == 1).nonzero(as_tuple=True)[0]
        k = min(500, len(idx0), len(idx1))
        sel = torch.cat((idx0[:k], idx1[:k]))
        X_test = (test.data[sel].float() / 255.0).to(device)
        y_test = test.targets[sel].float().to(device)

        return X_train, y_train, X_test, y_test