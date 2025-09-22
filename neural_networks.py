import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dimension=2, hidden_units=25, depth=5, vision_model=False, activation=nn.ReLU):
        """
        General constructor for feed-forward neural networks

        Args:
            input_dimension (int, optional): Input dimension. Defaults to 2.
            hidden_units (int, optional): Number of neurons in hidden layers. Defaults to 25.
            depth (int, optional): number of hidden layers. Defaults to 5.
            vision_model (bool, optional): If True, flatten input before forward pass. Defaults to False.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.vision_model = vision_model
        self.flatten = nn.Flatten()
        self.activation = activation()
        self.sigmoid = nn.Sigmoid()

        # First layer
        layers = [nn.Linear(input_dimension, hidden_units)]

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))

        # Output layer
        layers.append(nn.Linear(hidden_units, 1))

        # Store as ModuleList 
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input Tensor.

        Returns:
            torch.Tensor: Output tensor containing the raw predictions (before sigmoid).
        """
        if self.vision_model:
            x = self.flatten(x)

        # Pass through all but last layer with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # Final layer without sigmoid
        return self.layers[-1](x)

    def features(self, x):
        """
        Forward pass of the neural network while saving the neural features. 

        Args:
            x (torch.Tensor): Input Tensor.

        Returns:
            List[np.ndarray]: A list of Numpy arrays containing the features.
        """
        if self.vision_model:
            x = self.flatten(x)

        features = [x.detach().cpu().numpy()]

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            features.append(x.detach().cpu().numpy())

        x = self.sigmoid(self.layers[-1](x))
        features.append(x.detach().cpu().numpy())

        return features