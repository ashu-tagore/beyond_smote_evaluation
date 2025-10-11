"""
PyTorch deep learning models for imbalanced classification
Purpose: Implementing neural network architectures with batch normalization and weighted loss
"""

# Standard library imports
import sys
import time
from pathlib import Path

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Adding project modules to path (must be before local imports)
sys.path.append(str(Path(__file__).parent))

# Local imports (after path modification)
from config import SEED_VALUE, DL_CONFIG, PYTORCH_STORAGE


class BatchNormClassifier(nn.Module):
    """
    Neural network with batch normalization for binary classification

    Architecture:
        Input -> Linear -> BatchNorm -> ReLU -> Dropout ->
        Linear -> BatchNorm -> ReLU -> Dropout ->
        Linear -> BatchNorm -> ReLU -> Dropout ->
        Linear -> BatchNorm -> ReLU ->
        Linear -> Sigmoid -> Output
    """

    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        """
        Initializing batch-normalized neural network

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates for each hidden layer
        """
        super(BatchNormClassifier, self).__init__()

        # Setting default architecture if not provided
        if hidden_dims is None:
            hidden_dims = DL_CONFIG['architecture']

        if dropout_rates is None:
            dropout_rates = DL_CONFIG['dropout_vals']

        # Building layer list
        layers = []
        prev_dim = input_dim

        # Adding hidden layers with batch normalization
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))

            # ReLU activation
            layers.append(nn.ReLU())

            # Dropout (if not last hidden layer)
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))

            prev_dim = hidden_dim

        # Adding output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        # Creating sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through network

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        return self.model(x)


class WeightedLossClassifier(nn.Module):
    """
    Neural network using weighted loss for imbalanced classification

    Same architecture as BatchNormClassifier but designed to work with
    weighted BCE loss instead of data resampling
    """

    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        """
        Initializing weighted loss neural network

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rates: List of dropout rates for each hidden layer
        """
        super(WeightedLossClassifier, self).__init__()

        # Setting default architecture if not provided
        if hidden_dims is None:
            hidden_dims = DL_CONFIG['architecture']

        if dropout_rates is None:
            dropout_rates = DL_CONFIG['dropout_vals']

        # Building layer list
        layers = []
        prev_dim = input_dim

        # Adding hidden layers with batch normalization
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))

            # ReLU activation
            layers.append(nn.ReLU())

            # Dropout (if not last hidden layer)
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))

            prev_dim = hidden_dim

        # Adding output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        # Creating sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through network

        Args:
            x: Input tensor

        Returns:
            Output predictions
        """
        return self.model(x)


def calculate_pos_weight(y_train):
    """
    Calculating positive class weight for imbalanced data

    Args:
        y_train: Training labels

    Returns:
        Tensor containing positive weight
    """

    # Counting class occurrences
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    # Computing weight (ratio of negative to positive)
    pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    return torch.tensor([pos_weight], dtype=torch.float32)


def prepare_data_loaders(X_train, y_train, X_val, y_val, batch_size=None):
    """
    Preparing PyTorch data loaders for training and validation

    Args:
        X_train: Training features (numpy array or tensor)
        y_train: Training labels (numpy array or tensor)
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for training

    Returns:
        Tuple containing (train_loader, val_loader)
    """

    # Setting default batch size if not provided
    if batch_size is None:
        batch_size = DL_CONFIG['batch_count']

    # Converting to tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.FloatTensor(X_val)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.FloatTensor(y_val).reshape(-1, 1)

    # Creating datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, drop_last=False)

    return train_loader, val_loader


def train_pytorch_model(model, train_loader, val_loader, epochs=None,
                       learning_rate=None, pos_weight=None, patience=None,
                       device=None, verbose=True):
    """
    Training PyTorch model with early stopping

    Args:
        model: PyTorch model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        pos_weight: Weight for positive class (for weighted loss)
        patience: Early stopping patience
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress

    Returns:
        Tuple containing (trained_model, training_history)
    """

    # Setting default hyperparameters if not provided
    if epochs is None:
        epochs = DL_CONFIG['epoch_count']
    if learning_rate is None:
        learning_rate = DL_CONFIG['learn_rate']
    if patience is None:
        patience = DL_CONFIG['patience']
    if device is None:
        device = DL_CONFIG['compute_device']

    # Moving model to device
    device = torch.device(device)
    model = model.to(device)

    # Defining loss function
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCELoss(weight=None)  # Will apply pos_weight manually
    else:
        criterion = nn.BCELoss()

    # Defining optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initializing training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Recording start time
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            # Moving batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zeroing gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)

            # Computing loss
            if pos_weight is not None:
                # Applying positive weight manually
                loss = criterion(outputs, batch_y)
                weights = torch.where(batch_y == 1, pos_weight,
                                    torch.ones_like(batch_y))
                loss = (loss * weights).mean()
            else:
                loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulating metrics
            train_loss += loss.item() * batch_X.size(0)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Computing training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Moving batch to device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                outputs = model(batch_X)

                # Computing loss
                if pos_weight is not None:
                    loss = criterion(outputs, batch_y)
                    weights = torch.where(batch_y == 1, pos_weight,
                                        torch.ones_like(batch_y))
                    loss = (loss * weights).mean()
                else:
                    loss = criterion(outputs, batch_y)

                # Accumulating metrics
                val_loss += loss.item() * batch_X.size(0)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        # Computing validation metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Storing history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # Printing progress
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Val Acc: {epoch_val_acc:.4f}")

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Loading best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Computing total training time
    training_time = time.time() - start_time

    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")

    # Adding training time to history
    history['training_time'] = training_time
    history['best_val_loss'] = best_val_loss

    return model, history


def evaluate_pytorch_model(model, X_test, y_test, batch_size=None,
                          device=None, verbose=True):
    """
    Evaluating trained PyTorch model on test set

    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        verbose: Whether to print evaluation results

    Returns:
        Dictionary containing predictions and metrics
    """

    # Setting defaults
    if batch_size is None:
        batch_size = DL_CONFIG['batch_count']
    if device is None:
        device = DL_CONFIG['compute_device']

    # Converting to tensors if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    # Creating data loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)

    # Moving model to device
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Initializing prediction lists
    all_predictions = []
    all_probabilities = []

    # Recording prediction time
    start_time = time.time()

    # Evaluating
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)

            # Getting predictions
            outputs = model(batch_X)

            # Storing results
            all_probabilities.extend(outputs.cpu().numpy())
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())

    # Computing prediction time
    prediction_time = time.time() - start_time

    # Converting to numpy arrays
    y_pred = np.array(all_predictions).flatten()
    y_proba = np.array(all_probabilities).flatten()
    y_true = y_test.numpy().flatten()

    # Computing accuracy
    accuracy = (y_pred == y_true).mean()

    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Prediction time: {prediction_time:.4f} seconds")

    results = {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'prediction_time': prediction_time
    }

    return results


def save_pytorch_model(model, model_name, save_directory=None):
    """
    Saving trained PyTorch model to disk

    Args:
        model: Trained PyTorch model
        model_name: Name for saved model file
        save_directory: Directory to save model

    Returns:
        Path to saved model file
    """

    if save_directory is None:
        save_directory = PYTORCH_STORAGE

    # Creating directory if not existing
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Creating filename
    model_filename = f"{model_name}.pth"
    model_path = Path(save_directory) / model_filename

    # Saving model state dict
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

    return model_path


def load_pytorch_model(model_class, model_name, input_dim,
                      load_directory=None, device=None):
    """
    Loading trained PyTorch model from disk

    Args:
        model_class: Model class (BatchNormClassifier or WeightedLossClassifier)
        model_name: Name of saved model file
        input_dim: Number of input features
        load_directory: Directory containing model
        device: Device to load model on

    Returns:
        Loaded model instance
    """

    if load_directory is None:
        load_directory = PYTORCH_STORAGE
    if device is None:
        device = DL_CONFIG['compute_device']

    # Creating filename
    model_filename = f"{model_name}.pth"
    model_path = Path(load_directory) / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initializing model
    model = model_class(input_dim)

    # Loading state dict
    device = torch.device(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")

    return model


if __name__ == "__main__":
    print("="*70)
    print("PYTORCH MODELS MODULE TEST")
    print("="*70)

    # Setting random seed
    torch.manual_seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)

    # Generating synthetic data for testing
    n_samples = 1000
    n_features = 28

    X_train_test = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train_test = np.random.randint(0, 2, n_samples).astype(np.float32)

    X_val_test = np.random.randn(200, n_features).astype(np.float32)
    y_val_test = np.random.randint(0, 2, 200).astype(np.float32)

    print("\nGenerated synthetic dataset:")
    print(f"Training samples: {X_train_test.shape}")
    print(f"Validation samples: {X_val_test.shape}")

    print("\nTest 1: Initializing BatchNormClassifier...")
    model_bn = BatchNormClassifier(input_dim=n_features)
    print(f"Model created with {sum(p.numel() for p in model_bn.parameters())} parameters")

    print("\nTest 2: Calculating positive weight...")
    pos_weight_test = calculate_pos_weight(y_train_test)
    print(f"Positive weight: {pos_weight_test.item():.4f}")

    print("\nTest 3: Preparing data loaders...")
    train_loader_test, val_loader_test = prepare_data_loaders(
        X_train_test, y_train_test, X_val_test, y_val_test, batch_size=64
    )
    print(f"Train loader batches: {len(train_loader_test)}")
    print(f"Validation loader batches: {len(val_loader_test)}")

    print("\nTest 4: Training model (5 epochs)...")
    trained_model, history_test = train_pytorch_model(
        model_bn, train_loader_test, val_loader_test,
        epochs=5, patience=3, verbose=True
    )

    print("\nTest 5: Evaluating model...")
    eval_results_test = evaluate_pytorch_model(
        trained_model, X_val_test, y_val_test, verbose=True
    )

    print("\nTest 6: Initializing WeightedLossClassifier...")
    model_wl = WeightedLossClassifier(input_dim=n_features)
    print("Weighted loss model created")

    print("\nALL TESTS COMPLETED SUCCESSFULLY")
