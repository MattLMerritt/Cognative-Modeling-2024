
import torch
import torch.nn as nn

class ClientModel(nn.Module):
    def __init__(self, n_features, embeding_output_size, layers=1):
        super(ClientModel, self).__init__()

        # input layer
        self.fc1 = nn.Linear(n_features,  128)
        self.bn1 = nn.BatchNorm1d(128)

        # hidden layers (parameterized by layers param)
        layers = max(1, layers)
        self.linears = nn.ModuleList([nn.Linear(128, 128) for _ in range(layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(layers)])

        # hidden layers to embeding generation
        self.fc2 = nn.Linear(128, embeding_output_size)
        self.bn2 = nn.BatchNorm1d(embeding_output_size)

    def forward(self, x):
        # input layer
        x = torch.relu(self.bn1(self.fc1(x)))

        # hidden layers
        for i, l in enumerate(self.linears):
            x = l(x)
            x = torch.relu(self.bn[i](x))

        # embeding generation
        x = torch.tanh(self.bn2(self.fc2(x)))
        return x


class FusionModel(nn.Module):
    def __init__(self, n_embeding_components):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(n_embeding_components, 32)  # Adjust the input dimension as per concatenated input
        self.fc2 = nn.Linear(32, 1)

    def forward(self, combined_input):
        x = torch.relu(self.fc1(combined_input))
        x = torch.sigmoid(self.fc2(x))
        return x


def generateClientModels(feature_partitioning, embeding_size, layers=1):
    """Generate client models based on the feature partitioning and output embeding size.
    
    Parameters:
    -----------
    feature_partitioning     : list[int]
        The number of features per client.
    embeding_size : int
        The size of the embeding vector that each client should generate.

    Returns:
    --------
    x : list[ClientModel]
        An array containing client models paramertized based on function params.
    """
    models = []
    for _, input_features in enumerate(feature_partitioning):
        model = ClientModel(input_features, embeding_size, layers)
        models.append(model)
    return models


def generateFusionModel(n_embeding_components):
    """Generate a fusion model based on the number of embeding components.

    Parameters:
    -----------
    n_embeding_components : int
        The number of embeding components that the fusion model should expect.

    Returns:
    --------
    x : FusionModel
        A fusion model paramertized based on function params.   
    """
    return FusionModel(n_embeding_components)

