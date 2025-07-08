import torch
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from src.helpers import after_subplot
import numpy as np
import tempfile


def train_one_epoch(train_dataloader, model, optimizer, loss_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader, desc="Training", ncols=80)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += (loss.item() - running_loss) / (batch_idx + 1)

    return running_loss


def valid_one_epoch(valid_dataloader, model, loss_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(valid_dataloader, desc="Validating", ncols=80)):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = loss_fn(outputs, target)

            running_loss += (loss.item() - running_loss) / (batch_idx + 1)

    return running_loss


def one_epoch_test(test_dataloader, model, loss_fn):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader, desc="Testing", ncols=80)):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = loss_fn(outputs, target)

            test_loss += (loss.item() - test_loss) / (batch_idx + 1)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return test_loss, accuracy


def optimize(data_loaders, model, optimizer, loss_fn, n_epochs, save_path, interactive_tracking=False):
    best_val_loss = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=True)

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss_fn)
        val_loss = valid_one_epoch(data_loaders["valid"], model, loss_fn)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")

        if val_loss < best_val_loss * 0.99:  # Improved by >1%
            print(f"Validation loss improved from {best_val_loss:.4f} → {val_loss:.4f}. Saving model.")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        scheduler.step(val_loss)

        if interactive_tracking:
            logs = {"loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]['lr']}
            liveloss.update(logs)
            liveloss.send()

    
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

    test_loss, acc = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(test_loss), "Test loss is nan"
