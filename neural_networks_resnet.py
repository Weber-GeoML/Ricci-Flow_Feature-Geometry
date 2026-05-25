import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, input_dimension=2, hidden_units=25, depth=5, vision_model=False, activation=nn.ReLU):
        """
        Feed-forward neural network with residual (skip) connections.

        Each hidden layer applies x -> x + activation(Linear(x)), i.e., a pre-activation
        residual block. The first layer projects the input to the hidden dimension, and
        the final layer maps to a single output logit (no sigmoid).

        Args:
            input_dimension (int, optional): Input dimension. Defaults to 2.
            hidden_units (int, optional): Number of neurons in hidden layers. Defaults to 25.
            depth (int, optional): Number of hidden layers (residual blocks). Defaults to 5.
            vision_model (bool, optional): If True, flatten input before forward pass. Defaults to False.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.vision_model = vision_model
        self.flatten = nn.Flatten()
        self.activation = activation()
        self.sigmoid = nn.Sigmoid()

        # Projection layer (input_dimension -> hidden_units)
        self.input_layer = nn.Linear(input_dimension, hidden_units)

        # Residual hidden layers (hidden_units -> hidden_units)
        self.residual_layers = nn.ModuleList([
            nn.Linear(hidden_units, hidden_units) for _ in range(depth - 1)
        ])

        # Output layer (hidden_units -> 1)
        self.output_layer = nn.Linear(hidden_units, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Raw logits (before sigmoid).
        """
        if self.vision_model:
            x = self.flatten(x)

        # First layer (no skip connection, changes dimension)
        x = self.activation(self.input_layer(x))

        # Residual blocks: x -> x + activation(Linear(x))
        for layer in self.residual_layers:
            x = x + self.activation(layer(x))

        # Output layer
        return self.output_layer(x)

    def features(self, x):
        """
        Forward pass saving the feature representation after each layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[np.ndarray]: Feature representations at each layer
                              (input, after each hidden layer, output).
        """
        if self.vision_model:
            x = self.flatten(x)

        features = [x.detach().cpu().numpy()]

        # First layer
        x = self.activation(self.input_layer(x))
        features.append(x.detach().cpu().numpy())

        # Residual blocks
        for layer in self.residual_layers:
            x = x + self.activation(layer(x))
            features.append(x.detach().cpu().numpy())

        # Output layer with sigmoid
        x = self.sigmoid(self.output_layer(x))
        features.append(x.detach().cpu().numpy())

        return features
