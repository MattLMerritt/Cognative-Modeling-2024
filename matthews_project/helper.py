
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

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


def trainEnsemble(client_models, fusion_model, party_paritions, train_dataloader, client_optimizers=None, fusion_optimizer=None, loss_fn=nn.BCELoss(), epochs=10):
    """Train the clients and fusion model without adding noise to dataloader.

    Parameters:
    -----------
    client_models : list[nn.Module]
        A list of client models to train.
    fusion_model : nn.Module
        The fusion model to train.
    party_paritions : list[int]
        The number of features per party.
    train_dataloader : DataLoader
        The data loader that contains the training data.
    client_optimizers : list[Optimizer]
        The optimizers to use for training the client models.
    fusion_optimizer : Optimizer
        The optimizer to use for training the fusion model.
    loss_fn : Loss Function
        The loss function to use for training.
    epochs : int
        The number of epochs to train the model for.

    Returns:
    --------

     : float
        The average loss over the training data.
    """

    if(client_optimizers is None):
        client_optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]

    if(fusion_optimizer is None):
        fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)

    losses = []
    epo = []
    auprc_values = []

    epochs = 10


    # Training Loop
    for epoch in range(epochs):
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        for client_data_batch, batch_labels in train_dataloader:

            # re-construct batch data per party
            client_data_batched_per_party = torch.split(client_data_batch, party_paritions, dim=1)
            
            # Zero the gradients
            for optimizer in client_optimizers:
                optimizer.zero_grad()
            fusion_optimizer.zero_grad()
            
            # Independent Forward Pass
            client_outputs = []
            for i in range(len(party_paritions)):
                client_outputs.append(client_models[i](client_data_batched_per_party[i]))
            
            # Fusion Forward Pass
            predictions = fusion_model(torch.cat(client_outputs, dim=1))
            
            # Calculate Loss
            loss = loss_fn(predictions, batch_labels)
            total_loss += loss.item()
            
            # Backward pass and pptimize
            loss.backward()
            for optimizer in client_optimizers:
                optimizer.step()
            fusion_optimizer.step()

            # Store labels and predictions for metrics calculation
            all_labels.extend(batch_labels.detach().cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())
        
        # Metrics calculation
        avg_loss = total_loss / len(train_dataloader)
        auprc = average_precision_score(all_labels, all_predictions)
        
        losses.append(avg_loss)
        epo.append(epoch + 1)  # Add 1 to start epoch counting from 1
        auprc_values.append(auprc)  # Store AUPRC value

        print(f"Epoch {epoch + 1} - Loss: {avg_loss}, AUPRC: {auprc}")
    
    return (losses, epo, auprc_values)


def generateNoisyDataloader(dataset_tensors, noise_scale=0.1, multi_tensors=True, batch_size=32, shuffle=True):
    """Generate a noisy dataloader based on the input dataset tensors.

    Parameters:
    -----------
    dataset_tensors : list[Tensor]
        A list of tensors that represent the dataset.
    noise_scale : float
        The scale of the noise to add to the dataset.
    batch_size : int
        The batch size of the dataloader.
    shuffle : bool
        Whether to shuffle the dataloader.
    
    Returns:
    --------
    x : DataLoader
        A dataloader containing the noisy dataset.
    """

    if(multi_tensors):
        noisy_tensors = []
        for tensor in dataset_tensors:
            noisy_tensor = tensor + noise_scale * torch.randn_like(tensor)
            noisy_tensors.append(noisy_tensor)
        dataset = TensorDataset(*noisy_tensors)
    else:
        noisy_tensor = dataset_tensors + noise_scale * torch.randn_like(dataset_tensors)
        dataset = TensorDataset(noisy_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def generateLossGraph(losses, epochs, noise_scale=0.1):
    """Generate a loss graph based on the input losses and epochs.

    Parameters:
    -----------
    losses : list[float]
        A list of losses to plot.
    epochs : list[int]
        A list of epochs to plot.

    Returns:
    --------
    x : None
        A plot of the loss graph.
    """
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine(ax=ax)
    ax.plot(epochs, losses, marker='o', linestyle='-')
    plt.title(f'Loss vs. Epoch (Noise Scale: {noise_scale})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    return f

def generateAUPRCGraph(auprc_values, epochs, noise_scale=0.1):
    """Generate an AUPRC graph based on the input AUPRC values and epochs.

    Parameters:
    -----------
    auprc_values : list[float]
        A list of AUPRC values to plot.
    epochs : list[int]
        A list of epochs to plot.

    Returns:
    --------
    x : None
        A plot of the AUPRC graph.
    """
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine(ax=ax)
    ax.plot(epochs, auprc_values, marker='o', linestyle='-')
    plt.title(f'AUPRC vs. Epoch (Noise Scale: {noise_scale})')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.grid(True)
    plt.show()
    return f

def evaulateEnsemble(client_models, fusion_model, party_paritions, test_dataloader, loss_fn=nn.BCELoss()):
    """Evaluate the model based on the input test dataloader.

    Parameters:
    -----------
    client_models : list[nn.Module]
        A list of client models to evaluate.
    fusion_model : nn.Module
        The fusion model to evaluate.
    party_paritions : list[int]
        The number of features per party.
    test_dataloader : DataLoader
        The data loader that contains the test data.
    loss_fn : Loss Function
        The loss function to use for evaluation.

    Returns:
    --------
    x : float
        The average loss over the test data.
    """

    losses = []
    all_labels = []
    all_predictions = []

    for client_data_batch, batch_labels in test_dataloader:

        # re-construct batch data per party
        client_data_batched_per_party = torch.split(client_data_batch, party_paritions, dim=1)
        
        # Independent Forward Pass
        client_outputs = []
        for i in range(len(party_paritions)):
            client_outputs.append(client_models[i](client_data_batched_per_party[i]))
        
        # Fusion Forward Pass
        predictions = fusion_model(torch.cat(client_outputs, dim=1))
        
        # Calculate Loss
        loss = loss_fn(predictions, batch_labels)
        losses.append(loss.item())
        
        # Store labels and predictions for metrics calculation
        all_labels.extend(batch_labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())
    
    # Metrics calculation
    avg_loss = sum(losses) / len(test_dataloader)
    auprc = average_precision_score(all_labels, all_predictions)
    
    print(f"Loss: {avg_loss}, AUPRC: {auprc}")
    return avg_loss, auprc