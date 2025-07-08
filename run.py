#!/usr/bin/env python3
"""Simple interface for training neural networks on binary classification tasks."""

import torch as t
from circuits.mlp import SwiGLUMLP
from circuits.datasets import ParityDataset, PasswordParity, BinaryDataset
from circuits.trainer import Trainer, TrainingConfig


# Clean direct interface
def SwiGLUMLP(input_size: int, hidden_sizes: list[int] | None = None, output_size: int = 1, **kwargs):
    """Create SwiGLU MLP with sensible defaults.
    
    Args:
        input_size: Number of input bits
        hidden_sizes: Hidden layer sizes (default: [32, 16] for input_size >= 8, [16] for smaller)
        output_size: Number of outputs (default: 1 for binary classification)
    """
    from circuits.mlp import SwiGLUMLP as _SwiGLUMLP
    
    if hidden_sizes is None:
        if input_size >= 8:
            hidden_sizes = [32, 16]
        elif input_size >= 4:
            hidden_sizes = [16]
        else:
            hidden_sizes = [8]
    
    return _SwiGLUMLP(input_size, hidden_sizes, output_size, **kwargs)


def train(model, datagen: BinaryDataset, n_samples: int = 5000, lr: float = 1e-3, **kwargs):
    """Train model with continuous fresh data generation.
    
    Args:
        model: Neural network model
        datagen: Dataset generator (ParityDataset, PasswordParity, etc.)
        n_samples: Total number of training samples to generate
        lr: Learning rate
        **kwargs: Additional training config options
    
    Returns:
        Trained model
    """
    config = TrainingConfig(
        n_training_samples=n_samples,
        lr=lr,
        batch_size=kwargs.get('batch_size', 64),
        print_every=kwargs.get('print_every', max(500, n_samples // 10)),
        **{k: v for k, v in kwargs.items() if k not in ['batch_size', 'print_every']}
    )
    
    trainer = Trainer(model, datagen, config)
    trainer.run()
    return model


def test(model, datagen: BinaryDataset, n_samples: int = 100, seed: int = 999):
    """Test model on fresh examples.
    
    Args:
        model: Trained neural network model
        datagen: Dataset generator
        n_samples: Number of test samples to generate
        seed: Random seed for reproducible testing
    """
    model.eval()
    test_inputs, test_labels = datagen.generate_batch(n_samples, seed=seed)
    
    with t.no_grad():
        outputs = model(test_inputs)
        predictions = (t.sigmoid(outputs) > 0.5).float()
        
        correct = (predictions == test_labels).sum().item()
        accuracy = correct / n_samples
        
        print(f"Test accuracy: {accuracy:.4f} ({correct}/{n_samples})")
        
        # Show first few examples
        print("Sample predictions:")
        for i in range(min(5, n_samples)):
            input_bits = test_inputs[i].int().tolist()
            pred = int(predictions[i].item())
            expected = int(test_labels[i].item())
            status = "✓" if pred == expected else "✗"
            print(f"  {input_bits} -> {pred} (expected {expected}) {status}")
        
        return accuracy


def save(model, path: str):
    """Save model to file.
    
    Args:
        model: Trained neural network model
        path: File path to save to
    """
    t.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load(path: str, model_class=None, **model_kwargs):
    """Load model from file.
    
    Args:
        path: File path to load from
        model_class: Model class (default: SwiGLUMLP)
        **model_kwargs: Arguments for model constructor
    
    Returns:
        Loaded model
    """
    if model_class is None:
        model_class = SwiGLUMLP
    
    model = model_class(**model_kwargs)
    model.load_state_dict(t.load(path, map_location='cpu'))
    model.eval()
    return model


# Backward compatibility - existing interface
def train_model(dataset_name: str, **kwargs):
    """Train a model on the specified dataset with continuous fresh data generation.
    
    Args:
        dataset_name: 'parity' or 'password_parity'
        **kwargs: Optional parameters like bit_length, password, n_training_samples, lr, etc.
    """
    # Extract parameters with defaults
    bit_length = kwargs.get('bit_length', 8)
    n_training_samples = kwargs.get('n_training_samples', 5000)
    lr = kwargs.get('lr', 1e-3)
    hidden_sizes = kwargs.get('hidden_sizes', [32, 16])
    
    # Create dataset
    if dataset_name == 'parity':
        dataset = ParityDataset(bit_length=bit_length, n_samples=1000)  # Only used for validation
        print(f"Training on {bit_length}-bit parity with {n_training_samples:,} fresh samples")
        
    elif dataset_name == 'password_parity':
        password = kwargs.get('password', '101')
        dataset = PasswordParity(bit_length=bit_length, password=password, n_samples=1000)
        print(f"Training on {bit_length}-bit password parity (password: {password}) with {n_training_samples:,} fresh samples")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create model
    from circuits.mlp import SwiGLUMLP as _SwiGLUMLP
    model = _SwiGLUMLP(
        input_size=bit_length,
        hidden_sizes=hidden_sizes,
        output_size=1
    )
    
    # Configure training
    config = TrainingConfig(
        n_training_samples=n_training_samples,
        lr=lr,
        batch_size=64,
        print_every=max(500, n_training_samples // 10)  # Print 10 times during training
    )
    
    # Train
    trainer = Trainer(model, dataset, config)
    trainer.run()
    
    # Save model
    model_name = f"model_{dataset_name}_{bit_length}bit.pth"
    trainer.save_model(model_name)
    
    return trainer


def test_model(model_path: str, dataset_name: str, **kwargs):
    """Test a saved model on some examples."""
    # Load model
    model = Trainer.load_model(model_path)
    
    # Generate test examples
    bit_length = kwargs.get('bit_length', 8)
    
    if dataset_name == 'parity':
        test_data = ParityDataset(bit_length=bit_length, n_samples=1)
        test_inputs, test_labels = test_data.generate_batch(5, seed=999)
    elif dataset_name == 'password_parity':
        password = kwargs.get('password', '101')
        test_data = PasswordParity(bit_length=bit_length, password=password, n_samples=1)
        test_inputs, test_labels = test_data.generate_batch(5, seed=999)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Test predictions
    print(f"\nTesting model on {dataset_name} examples:")
    model.eval()
    
    with t.no_grad():
        outputs = model(test_inputs)
        predictions = (t.sigmoid(outputs) > 0.5).float()
        
        for i in range(len(test_inputs)):
            input_bits = test_inputs[i].int().tolist()
            pred = int(predictions[i].item())
            expected = int(test_labels[i].item())
            status = "✓" if pred == expected else "✗"
            print(f"  Input: {input_bits} -> Predicted: {pred}, Expected: {expected} {status}")


if __name__ == "__main__":
    print("=== Circuits Training Interface (Continuous Data Generation) ===\n")
    
    # Example 1: Train on parity with continuous data
    print("Example 1: Training on parity dataset")
    trainer1 = train_model('parity', bit_length=6, n_training_samples=3000, lr=1e-3)
    test_model('model_parity_6bit.pth', 'parity', bit_length=6)
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Train on password parity with continuous data
    print("Example 2: Training on password parity dataset")
    trainer2 = train_model('password_parity', bit_length=8, password='110', n_training_samples=4000, lr=1e-3)
    test_model('model_password_parity_8bit.pth', 'password_parity', bit_length=8, password='110')
    
    print(f"\nTraining complete! Models saved.")
    print(f"Parity final accuracy: {trainer1.history['val_acc'][-1]:.4f}")
    print(f"Password parity final accuracy: {trainer2.history['val_acc'][-1]:.4f}") 