import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one epoch of training.

    Args:
        train_dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        loss (Loss): The loss function.

    Returns:
        float: The average training loss for the epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    running_loss = 0.0 

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        data, target = data.to(device), target.to(device)
      
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        running_loss += (loss_value.item() - running_loss) / (batch_idx + 1)

    return running_loss 


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validates the model at the end of one epoch.

    Args:
        valid_dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): The neural network model.
        loss (Loss): The loss function.

    Returns:
        float: The average validation loss for the epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)
        

            output = model(data)
            loss_value = loss(output, target)

            running_loss += (loss_value.item() - running_loss) / (batch_idx + 1)

    return running_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    """
    Optimizes the model.

    Args:
        data_loaders (dict): Dictionary containing train, validation, and test data loaders.
        model (nn.Module): The neural network model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        loss (Loss): The loss function.
        n_epochs (int): Number of epochs for training.
        save_path (str): Path to save the trained model.
        interactive_tracking (bool, optional): Whether to use interactive loss tracking. Defaults to False.
    """
    
    valid_loss2 = float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    # Learning rate scheduler: reduce the learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,verbose=True)

    for epoch in range(1, n_epochs + 1):
        # Training phase
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)

        # Validation phase
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # Print training/validation statistics
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")

        if valid_loss < valid_loss2 * 0.99:  
            print(f"Validation loss improved from {valid_loss2:.2f} → {valid_loss:.2f}. Saving model.")
            valid_loss2 = valid_loss
            torch.save(model.state_dict(), save_path)

        # Update learning rate based on validation loss
        scheduler.step(valid_loss)

        # Log the losses and current learning rate
        if interactive_tracking:
            logs={"loss": train_loss, "val_loss": valid_loss ,"lr": optimizer.param_groups[0]["lr"]}
            liveloss.update(logs)
            liveloss.send()

    # Load the last checkpoint with the best model
    #model.load_state_dict(torch.load(save_path))


def one_epoch_test(test_dataloader, model, loss):
    """
    Tests the model for one epoch.

    Args:
        test_dataloader (DataLoader): DataLoader for test data.
        model (nn.Module): The neural network model.
        loss (Loss): The loss function.

    Returns:
        float: The average test loss for the epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            desc='Testing',
            total=len(test_dataloader),
            leave=True,
            ncols=80
        ):
            data, target = data.to(device), target.to(device)
          

            output = model(data)
            loss_value = loss(output, target)

            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.item() - test_loss))

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %4d%% (%2d/%2d)' % (
        100.0 * correct / total, correct, total))

    return test_loss

######################################################################################
#                                     TESTS
######################################################################################

import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"