
# Libraries for Data Handling
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Libraries for Algorithms
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, accuracy_score

# Libraries for Data Visulation Tools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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


def trainEnsemble(client_models, fusion_model, party_paritions, train_dataloader, test_dataloader, client_optimizers=None, fusion_optimizer=None, loss_fn=nn.BCEWithLogitsLoss(), epochs=10):
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

    training_losses = []
    epo = []
    training_auprc_values = []

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
        
        training_losses.append(avg_loss)
        epo.append(epoch + 1)  # Add 1 to start epoch counting from 1
        training_auprc_values.append(auprc)  # Store AUPRC value

        print(f"Epoch {epoch + 1} - Training Loss: {avg_loss}, AUPRC: {auprc}")

        # Metrics over test set
        evaluation_params = {
        "client_models": client_models,
        "fusion_model": fusion_model,
        "party_paritions": party_paritions,
        "test_dataloader": test_dataloader,
        "loss_fn": loss_fn,
    }
        test_loss, test_auprc = evaulateEnsemble(**evaluation_params)

    return (training_losses, epo, training_auprc_values)


def generateNoisyDataloader(dataset_tensors, label_tensor, noise_scale=0.1, batch_size=32, shuffle=True):
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

    concated_tensor = torch.cat(dataset_tensors, dim=1)
    noisy_tensor = concated_tensor + noise_scale * torch.randn_like(concated_tensor)
    noisy_dataset = TensorDataset(noisy_tensor, label_tensor)
    dataloader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def generateLossGraph(losses, epochs, n_distinct_parties, noise_scale=0.1, trained_with_noise=False):
    """Generate a loss graph based on the input losses and epochs.

    Parameters:
    -----------
    losses : list[float]
        A list of losses to plot.
    epochs : list[int]
        A list of epochs to plot.
    n_distinct_parties : int
        The number of distinct parties in the dataset.
    trained_with_noise : bool
        Whether the model was trained with noise.

    Returns:
    --------
    x : None
        A plot of the loss graph.
    """
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine(ax=ax)
    ax.plot(epochs, losses, marker='o', linestyle='-')
    plt.title(f'Loss vs. Epoch (Noise Scale: {noise_scale})' + (' (Trained with Noise)' if trained_with_noise else '') )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"results/adult_{n_distinct_parties}p/loss_{noise_scale}.png")
    plt.show()
    return f


def generateAUPRCGraph(auprc_values, epochs, n_distinct_parties, noise_scale=0.1, trained_with_noise=False):
    """Generate an AUPRC graph based on the input AUPRC values and epochs.

    Parameters:
    -----------
    auprc_values : list[float]
        A list of AUPRC values to plot.
    epochs : list[int]
        A list of epochs to plot.
    n_distinct_parties : int
        The number of distinct parties in the dataset.
    trained_with_noise : bool
        Whether the model was trained with noise.

    Returns:
    --------
    x : None
        A plot of the AUPRC graph.
    """
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine(ax=ax)
    ax.plot(epochs, auprc_values, marker='o', linestyle='-')
    plt.title(f'AUPRC vs. Epoch (Noise Scale: {noise_scale})' + (' (Trained with Noise)' if trained_with_noise else '') )
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.grid(True)
    plt.savefig(f"results/adult_{n_distinct_parties}p/auprc_{noise_scale}.png")
    plt.show()
    return f


def generateTotalLossMetricsGraph(losses, epochs, n_distinct_parties, noise_scale, hasMultipleNoises=False):
    """Generate a total loss graph based on the input losses and epochs.

    Parameters:
    -----------
    losses : dict[str](list[float])
        A dictionary of list of losses to plot, the strinf is the label.
    epochs : dict[str](list[float])
        A list of epochs to plot.
    n_distinct_parties : int
        The number of distinct parties in the dataset.
    trained_with_noise : bool
        Whether the model was trained with noise.
    hasMultipleNoises : bool
        Whether the aurpc spans multiple noise scales.

    Returns:
    --------
    x : Figure
        A plot of the loss graph.
    """
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine(ax=ax)
    for label in losses.keys():
        ax.plot(epochs, losses[label], marker='o', linestyle='-', label=label)
    
    if(hasMultipleNoises):
        plt.title(f'Loss vs. Epoch Over Multiple Noise Scales')
    else:
        plt.title(f'Loss vs. Epoch Over {noise_scale} Noise Scale')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    if(hasMultipleNoises):
        plt.savefig(f"results/adult_{n_distinct_parties}p/total_loss_over_multiple_noises.png")
    else:
        plt.savefig(f"results/adult_{n_distinct_parties}p/total_loss_over_{noise_scale}_noise_scale.png")
    plt.show()
    return f


def generateTotalAuprcMetricsGraph(aurpc, epochs, n_distinct_parties, noise_scale, hasMultipleNoises=False):
    """Generate a total loss graph based on the input losses and epochs.

    Parameters:
    -----------
    auprc : dict[str](list[float])
        A dictionary of list of losses to plot, the strinf is the label.
    epochs : dict[str](list[float])
        A list of epochs to plot.
    n_distinct_parties : int
        The number of distinct parties in the dataset.
    trained_with_noise : bool
        Whether the model was trained with noise.
    hasMultipleNoises : bool
        Whether the aurpc spans multiple noise scales.

    Returns:
    --------
    f : Figure
        A plot of the loss graph.
    """
    
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine(ax=ax)
    for label in aurpc.keys():
        ax.plot(epochs, aurpc[label], marker='o', linestyle='-', label=label)
    if(hasMultipleNoises):
        plt.title(f'AUPRC vs. Epoch Over Multiple Noise Scales')
    else:
        plt.title(f'AUPRC vs. Epoch Over {noise_scale} Noise Scale')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.grid(True)
    plt.legend()
    if(hasMultipleNoises):
        plt.savefig(f"results/adult_{n_distinct_parties}p/total_auprc_over_multiple_noises.png")
    else:
        plt.savefig(f"results/adult_{n_distinct_parties}p/total_auprc_over_{noise_scale}_noise_scale.png")
    plt.show()
    return f

def evaulateEnsemble(client_models, fusion_model, party_paritions, test_dataloader, generate_confusion=False, loss_fn=nn.BCELoss()):
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
    avg_loss : float
        The average loss over the test data.
    auprc : float
        The AUPRC score over the test data.
    """

    threshold = 0.5
    losses = []
    all_test_binary_predictions = []
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
        binary_predictions = (predictions.sigmoid().detach().cpu().numpy() > threshold).astype(int).flatten()
        all_test_binary_predictions = np.concatenate((all_test_binary_predictions, binary_predictions))
        
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

    if(generate_confusion):
        # generate the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_test_binary_predictions)

        # Plotting the confusion matrix using matplotlib
        fig, ax = plt.subplots()
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        for (i, j), val in np.ndenumerate(conf_matrix):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Adjust the ticks to show the corresponding labels
        ax.set_xticks(range(len(np.unique(all_labels))))
        ax.set_yticks(range(len(np.unique(all_labels))))
        ax.set_xticklabels(np.unique(all_labels))
        ax.set_yticklabels(np.unique(all_labels))

        plt.savefig(f"results/adult_{len(party_paritions)}p/confusion_matrix.png")
        plt.show()

    return avg_loss, auprc


def trainAndEvaulateEnsemble(party_paritions, train_dataloader, test_dataloader, layers, noise_scale, trained_with_noise=False, client_optimizers=None, fusion_optimizer=None, epochs=10):
    """Train and evaluate the ensemble model based on the input parameters.
    
    Parameters:
    -----------
    party_paritions : list[int]
        The number of features per party.
    train_dataloader : DataLoader
        The data loader that contains the training data.
    test_dataloader : DataLoader
        The data loader that contains the test data.
    layers : int
        The number of layers in the client models.
    noise_scale : float
        The scale of the noise to add to the dataloader.
    trained_with_noise : bool
        Whether the model was trained with noise.
    client_optimizers : list[Optimizer]
        The optimizers to use for training the client models.
    fusion_optimizer : Optimizer
        The optimizer to use for training the fusion model.
    epochs : int
        The number of epochs to train the model for.

    Returns:
    --------
    x : tuple
        A tuple containing the losses, epochs, and AUPRC values.
    """

    ## define models
    client_output_embedding_size = 16
    client_models = generateClientModels(party_paritions, client_output_embedding_size, layers)

    fusion_model = generateFusionModel(client_output_embedding_size * len(party_paritions))


    # define optimizers and loss function
    if(client_optimizers is None):
        client_optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in client_models]
    if(fusion_optimizer is None):
        fusion_optimizer = optim.SGD(fusion_model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()


    ## Train the ensemble model without noise
    training_params = {
        "client_models": client_models,
        "fusion_model": fusion_model,
        "party_paritions": party_paritions,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "client_optimizers": client_optimizers,
        "fusion_optimizer": fusion_optimizer,
        "loss_fn": loss_fn,
        "epochs": epochs
    }

    losses, epo, auprc_values = trainEnsemble(**training_params)

    ## Evaluate the ensemble model
    generateLossGraph(losses, epo, len(party_paritions), noise_scale, trained_with_noise)
    generateAUPRCGraph(auprc_values, epo, len(party_paritions), noise_scale, trained_with_noise)

    return (losses, epo, auprc_values)



def generateAnalysis(n_distinct_parties, layers, noise_scales=[0.0], training_epochs=10, batch_size=32, shuffle=True):
    """Generate the analysis based on the input parameters.

    Parameters:
    -----------
    n_distinct_parties : int
        The number of distinct parties in the dataset.
    layers : int
        The number of layers in the client models.

    Returns:
    --------
    x : None
        The results of the analysis.

    Notes:
    ------
    the following directories must be created before running the function:
    - datasets/pre_processed_adult_{n_distinct_parties}p
    """

    # dataset dir (must be generated by pre-processor for respective n_distinct_parties)
    dataset_dir = f"datasets/pre_processed_adult_{n_distinct_parties}p"

    # results dir (where to save the results)
    results_dir = f"results/adult_{n_distinct_parties}p"

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    ## Import pre-processed Dataset
    test_datframes = []
    train_dataframes = []
    for i in range(1, n_distinct_parties+1):
        test_datframes.append(pd.read_csv(f"{dataset_dir}/test_data_party_{i}.csv", sep=',', header=0))
        train_dataframes.append(pd.read_csv(f"{dataset_dir}/train_data_party_{i}.csv", sep=',', header=0))

    test_y_dataframe = pd.read_csv(f"{dataset_dir}/test_labels.csv", sep=',', header=0)
    train_y_dataframe = pd.read_csv(f"{dataset_dir}/train_labels.csv", sep=',', header=0)

    # Convert DataFrames to Tensors
    test_tensors = []
    train_tensors = []
    for i in range(n_distinct_parties):
        test_tensors.append(torch.tensor(test_datframes[i].values, dtype=torch.float32))
        train_tensors.append(torch.tensor(train_dataframes[i].values, dtype=torch.float32))

    test_y = torch.tensor(test_y_dataframe.values, dtype=torch.float32)
    train_y = torch.tensor(train_y_dataframe.values, dtype=torch.float32)

    # identify feature->party parition:
    party_paritions = [len(i.columns) for i in train_dataframes]

    ## Create dataset/dataloader for party input (concatonate all parties for clean dataloader)
    test_y = torch.tensor(test_y_dataframe.values, dtype=torch.float32)
    train_y = torch.tensor(train_y_dataframe.values, dtype=torch.float32)

    # Create dataset
    train_dataset = TensorDataset(torch.cat(train_tensors, dim=1), train_y)
    test_dataset = TensorDataset(torch.cat(test_tensors, dim=1), test_y)

    # Create dataloader
    train_dataloader_without_noise = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # test_dataloader_without_noise = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Generate noisy dataloaders
    noisy_dataloader_params = {
        "dataset_tensors": train_tensors,
        "label_tensor": train_y,
        "noise_scale": noise_scales[0],
        "batch_size": batch_size,
        "shuffle": shuffle
    }
    noisy_train_dataloaders = []
    noisy_test_dataloaders = []
    for noise_scale in noise_scales:
        noisy_dataloader_params["noise_scale"] = noise_scale
        noisy_train_dataloaders.append(generateNoisyDataloader(**noisy_dataloader_params))
        noisy_test_dataloaders.append(generateNoisyDataloader(**noisy_dataloader_params))
    
    # Deploy the ensemble model with various noise scales
    all_noise_loss = dict()
    all_noise_auprc = dict()
    for i, noise_scale in enumerate(noise_scales):
        print(f"Training Ensemble Model with Noise Scale: {noise_scale}")
        noisy_train_dataloader = noisy_train_dataloaders[i]
        noisy_test_dataloader = noisy_test_dataloaders[i]

        train_and_evaluate_params = {
            "party_paritions": party_paritions,
            "train_dataloader": noisy_train_dataloader,
            "test_dataloader": noisy_test_dataloader,
            "layers": layers,
            "noise_scale": noise_scale,
            "trained_with_noise": True,
            "epochs": training_epochs
        }

        local_noise_loss = dict()
        local_noise_auprc = dict()

        ## train with noise:
        losses, epo, auprc_values = trainAndEvaulateEnsemble(**train_and_evaluate_params)
        local_noise_loss[f"noise={noise_scale} nt"] = losses
        local_noise_auprc[f"noise={noise_scale} nt"] = auprc_values

        # Deploy the ensemble model with no noise during training
        train_and_evaluate_params["train_dataloader"] = train_dataloader_without_noise
        losses, epo, auprc_values = trainAndEvaulateEnsemble(**train_and_evaluate_params)
        local_noise_loss[f"noise={noise_scale} wnt"] = losses
        local_noise_auprc[f"noise={noise_scale} wnt"] = auprc_values


        # generate loss & auprc graph for current noises loss
        print("About to run: generateTotalLossMetricsGraph")
        generateTotalLossMetricsGraph(local_noise_loss, epo, len(party_paritions), noise_scale)
        generateTotalAuprcMetricsGraph(all_noise_auprc, epo, len(party_paritions), noise_scale)
        
        # add local loss and auprc values to global
        all_noise_loss.update(local_noise_loss)
        all_noise_auprc.update(local_noise_auprc)

    # generate total loss graph
    generateTotalLossMetricsGraph(all_noise_loss, epo, len(party_paritions), None, True)
    generateTotalAuprcMetricsGraph(all_noise_auprc, epo, len(party_paritions), None, True)


    return None


if __name__ == '__main__':
    print('running analysis for adult dataset')

    # 3 party data partition
    analysis_params = {
        "n_distinct_parties": 3,
        "layers": 1,
        "noise_scales": [0.01, 0.1, 0.3, 0.5, 1, 2, 5],
        "training_epochs": 10,
        "batch_size": 32,
    }
    generateAnalysis(**analysis_params)
    print('completed analsis for 3 party data partition')

    # 7 party data partition
    analysis_params["n_distinct_parties"] = 7
    generateAnalysis(**analysis_params)
    print('completed analsis for 7 party data partition')

