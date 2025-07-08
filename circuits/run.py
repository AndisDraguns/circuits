#!/usr/bin/env python3
"""Simple interface for training neural networks on binary classification tasks."""

import torch as t
from circuits.mlp import SwiGLUMLP
from circuits.datasets import ParityDataset, PasswordParity
from circuits.trainer import Trainer, TrainingConfig


def train_model(dataset_name: str, **kwargs):
    """Train a model on the specified dataset with optional parameters.
    
    Args:
        dataset_name: 'parity' or 'password_parity'
        **kwargs: Optional parameters like bit_length, password, epochs, lr, etc.
    """
    # Extract parameters with defaults
    bit_length = kwargs.get('bit_length', 8)
    epochs = kwargs.get('epochs', 50)
    lr = kwargs.get('lr', 1e-3)
    hidden_sizes = kwargs.get('hidden_sizes', [32, 16])
    n_samples = kwargs.get('n_samples', 5000)
    
    # Create dataset
    if dataset_name == 'parity':
        dataset = ParityDataset(bit_length=bit_length, n_samples=n_samples)
        print(f"Training on {bit_length}-bit parity dataset")
        
    elif dataset_name == 'password_parity':
        password = kwargs.get('password', '101')
        dataset = PasswordParity(bit_length=bit_length, password=password, n_samples=n_samples)
        print(f"Training on {bit_length}-bit password parity dataset (password: {password})")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create model
    model = SwiGLUMLP(
        input_size=bit_length,
        hidden_sizes=hidden_sizes,
        output_size=1
    )
    
    # Configure training
    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        batch_size=64,
        print_every=10
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
    
    # Create test examples
    bit_length = kwargs.get('bit_length', 8)
    
    if dataset_name == 'parity':
        test_data = ParityDataset(bit_length=bit_length, n_samples=5, seed=999)
    elif dataset_name == 'password_parity':
        password = kwargs.get('password', '101')
        test_data = PasswordParity(bit_length=bit_length, password=password, n_samples=5, seed=999)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Test predictions
    print(f"\nTesting model on {dataset_name} examples:")
    model.eval()
    
    with t.no_grad():
        for i in range(min(5, len(test_data))):
            inputs, expected = test_data[i]
            output = model(inputs.unsqueeze(0))
            prediction = (t.sigmoid(output) > 0.5).float()
            
            input_bits = inputs.int().tolist()
            print(f"  Input: {input_bits} -> Predicted: {int(prediction.item())}, Expected: {int(expected.item())}")


if __name__ == "__main__":
    print("=== Circuits Training Interface ===\n")
    
    # # Example 1: Train on parity
    # print("Example 1: Training on parity dataset")
    # trainer1 = train_model('parity', bit_length=6, epochs=30, lr=1e-3)
    # test_model('model_parity_6bit.pth', 'parity', bit_length=6)
    
    # print("\n" + "="*50 + "\n")
    
    # Example 2: Train on password parity
    print("Example 2: Training on password parity dataset")
    trainer2 = train_model('password_parity', bit_length=16, password='110110', epochs=40, lr=1e-3)
    test_model('model_password_parity_8bit.pth', 'password_parity', bit_length=8, password='110')
    
    print("\nTraining complete! Check the .pth files for saved models.") 