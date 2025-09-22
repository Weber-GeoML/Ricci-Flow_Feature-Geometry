import torch
import torch.nn as nn
from ricci_coefficients import Ricci_Coefficients

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between labels and predictions.

    Args:
        y_true (torch.Tensor): Labels of samples.
        y_pred (torch.Tensor): Predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 99.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



def train_model(threshold_accuracy, model, X_train, y_train, X_test, y_test, max_epochs=20000, verbose=False):
    """
    Trains a binary classification PyTorch model until a desired training accuracy is reached or a maximum number of epochs is exceeded.

    Args:
        threshold_accuracy (float): Target training accuracy (in percent).
        model (torch.nn.Module): The PyTorch model to be trained.
        X_train (torch.Tensor): Train data of shape (num_samples, num_features).
        y_train (torch.Tensor): Train labels of shape (num_samples,).
        X_test (torch.Tensor): Test data of shape (num_samples, num_features).
        y_test (torch.Tensor): Test labels of shape (num_samples,).
        max_epochs (int, optional): Maximum number of epochs. Defaults to 10000.
        verbose (bool, optional): Print out information. Defaults to False
    """
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    ### Training
    acc = 0
    epochs = 0
    while epochs < max_epochs:
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))

        # 2. Calculate the loss and accuracy
        loss = loss_fn(y_logits, y_train) 
        acc = accuracy_fn(y_train, y_preds)
        if acc >= threshold_accuracy:
            break
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        epochs +=1

    ### Testing
    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test accuracy
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)
    if verbose:
        print(f"Training finished | Epochs: {epochs} | Train acc: {acc:.2f}% | Test acc: {test_acc:.2f}%")

    return test_acc


def train_model_with_ricci_coefs(epochs, model, X_train, y_train, X_test, y_test, calculate_ricci_coefs_every=1, k=50, curv='Ollivier-Ricci', verbose=False):
    """
    Trains a binary classification PyTorch model and periodically computes the train and test accuracy as well as the local Ricci coefficients.

    Args:
        epochs (int): Number of epochs.
        model (torch.nn.Module): PyTorch model.
        X_train (torch.Tensor): Train data.
        y_train (torch.Tensor): Train labels.
        X_test (torch.Tensor): Test data.
        y_test (torch.Tensor): Test labels.
        calculate_ricci_coefs_every (int, optional): Frequency at which to save feature representations. Defaults to 1.
        k (int, optional): Number of neighbors in k-nearest-neighbor graph. Defaults to 50.
        curv (str, optional): Curvature notion. Defaults to "Ollivier-Ricci".
        verbose (bool, optional): Print out information. Defaults to False

    Returns:
        train_accuracies (List[float]): Training accuracies recorded at each saved epoch.
        test_accuracies (List[float]): Test accuracies recorded at each saved epoch.
        ricci_coefficients (List[np.array]): Local Ricci coefficients.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    ricci_coefficients = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        # --- Evaluation and local Ricci coefficients ---
        if epoch % calculate_ricci_coefs_every == 0:
            model.eval()
            with torch.inference_mode():
                # 1. Forward pass
                test_logits = model(X_test).squeeze()
                test_preds = torch.round(torch.sigmoid(test_logits))
                train_logits = model(X_train).squeeze()
                train_preds = torch.round(torch.sigmoid(train_logits))
                train_acc = accuracy_fn(y_train, train_preds)
                test_acc = accuracy_fn(y_test, test_preds)

                ricci_coefficients.append(Ricci_Coefficients(
                    model, X_test, k
                ).local_ricci_coefficient(curv=curv))
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            if verbose:
                print(f"Epoch: {epoch} | Train accuracy: {train_acc:.2f}% | Test accuracy: {test_acc:.2f}% ")


        # --- Training ---
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()

        # 2. Calculate the loss
        loss = loss_fn(y_logits, y_train) 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    return train_accuracies, test_accuracies, ricci_coefficients